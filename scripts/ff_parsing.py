"""
Class definitions for reading, writing, modifying, reducing, and merging
LAMMPS forcefields.

This is a minimal, class-based rewrite of the original procedural script.
Three classes are provided:

    Style       - one *_style category (pair/bond/angle/dihedral/improper)
    Coeff       - one *_coeff line (a single type, or a type pair for "pair")
    Forcefield  - owns a list of Styles and a list of Coeffs, and knows how
                  to read, merge, reduce, clone, and write them.

self.styles is a flat list[Style] (one per style_type actually present).
self.coeffs is a flat list[Coeff], always kept sorted by
(style_type, i, j) so writers/mergers/reducers never need to re-sort.
"""

from dataclasses import dataclass, field
from lammps_data import LammpsData

STYLE_TYPES = ("pair", "bond", "angle", "dihedral", "improper")


# --------------------------------------------------------------------------
# Small stand-alone parsing helpers (kept as functions since they don't hold
# state and are shared by multiple classes/methods).
# --------------------------------------------------------------------------

def _try_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def _try_int(tok):
    try:
        return int(tok)
    except (TypeError, ValueError):
        return None


def _tokens_equal(a, b):
    a_parts, b_parts = a.split(), b.split()
    if len(a_parts) != len(b_parts):
        return False
    for x, y in zip(a_parts, b_parts):
        fx, fy = _try_float(x), _try_float(y)
        if fx is not None and fy is not None:
            if fx != fy:
                return False
        elif x != y:
            return False
    return True


def _strip_accelerator_suffix(name):
    for suffix in ("/kk", "/omp"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def _split_hybrid_style(style_str):
    """Split a hybrid style into substyles and scale factors.

    Returns:
        (substyles, scale_factors)

    For hybrid/overlay:
        'hybrid/overlay lj/cut 1.0 lj/cut/soft 1 0.5'
        ->
        (['lj/cut 1.0', 'lj/cut/soft 1 0.5'], [None, None])

    For hybrid/scaled:
        'hybrid/scaled 0.5 lj/cut 1.0 1.0 lj/cut/soft 1 0.5'
        ->
        (['lj/cut 1.0', 'lj/cut/soft 1 0.5'], [0.5, 1.0])

    Scale factors may be numeric or LAMMPS variable expressions, e.g.
        'v_lambda'
        '1.0-v_lambda'
    """
    parts = style_str.strip().split()
    if not parts or "hybrid" not in parts[0]:
        return [style_str.strip()], [None]

    hybrid_style = parts[0]
    parts = parts[1:]

    if hybrid_style == "hybrid/scaled":
        substyles = []
        scale_factors = []
        current = []

        # A hybrid/scaled entry begins with: scale_factor substyle args
        #
        # so the scale factor tells us where each substyle begins.
        #
        # We need to identify the substyle token, which is the first
        # non-numeric token after the scale factor. Once the substyle
        # starts, numeric tokens belong to its arguments until the next
        # scale factor + substyle pair.

        i = 0
        while i < len(parts):
            # First token is the scale factor.
            scale_factor = parts[i]
            i += 1

            current = []

            # Find the substyle name: first non-numeric token.
            while i < len(parts):
                part = parts[i]
                if _try_float(part) is None:
                    current.append(part)
                    i += 1
                    break
                current.append(part)
                i += 1

            if not current:
                raise ValueError(
                    f"Invalid hybrid/scaled style: missing substyle after "
                    f"scale factor {scale_factor!r}"
                )

            # Collect arguments until we encounter what looks like the
            # beginning of the next scale_factor + substyle pair.
            #
            # Since scale factors and style arguments can both be numeric,
            # we use the presence of a non-numeric token to identify the
            # next substyle. The current substyle name is already the first
            # non-numeric token, so subsequent non-numeric tokens are
            # ambiguous without knowing the argument structure.
            #
            # Therefore, use the same grouping convention as the original
            # parser: a non-numeric token starts a new substyle.
            while i < len(parts):
                part = parts[i]

                if _try_float(part) is None:
                    break

                current.append(part)
                i += 1

            substyles.append(" ".join(current))
            scale_factors.append(scale_factor)

        # Convert numeric scale factors to floats while preserving LAMMPS
        # variable/string expressions.
        scale_factors = [
            _try_float(factor) if _try_float(factor) is not None else factor
            for factor in scale_factors
        ]

        return substyles, scale_factors

    # hybrid/overlay, hybrid/molecular, etc.
    substyles = []
    current = []

    for part in parts:
        if _try_float(part) is None:
            if current:
                substyles.append(" ".join(current))
            current = [part]
        else:
            current.append(part)

    if current:
        substyles.append(" ".join(current))

    return substyles, [None] * len(substyles)

from pathlib import Path


def _iter_input_lines(filename, _include_stack=None):
    """Yield logical, comment-free lines from a parameter file and its includes.

    Includes are resolved relative to the file containing the include
    statement. Lines ending in '&' are joined with the following line.
    """

    filename = Path(filename).resolve()

    if _include_stack is None:
        _include_stack = []

    if filename in _include_stack:
        chain = " -> ".join(str(p) for p in _include_stack + [filename])
        raise ValueError(f"Include cycle detected: {chain}")

    _include_stack.append(filename)

    try:
        with open(filename) as f:
            pending = ""

            for raw_line in f:
                # Remove comments before checking continuation.
                line = raw_line.split("#", 1)[0].strip()

                if not line:
                    continue

                if line.endswith("&"):
                    pending += line[:-1].rstrip() + " "
                    continue

                line = pending + line
                pending = ""

                parts = line.split()

                if not parts:
                    continue

                if parts[0].lower() == "include":
                    if len(parts) != 2:
                        raise ValueError(
                            f"{filename}: malformed include: {line}"
                        )

                    include_file = Path(parts[1])
                    if not include_file.is_absolute():
                        include_file = filename.parent / include_file

                    yield from _iter_input_lines(
                        include_file,
                        _include_stack,
                    )
                    continue

                yield line

            if pending:
                raise ValueError(
                    f"{filename}: continuation '&' at end of file "
                    "without a following line"
                )

    finally:
        _include_stack.pop()


def _coeff_sort_key(coeff):
    """Sort key used to keep Forcefield.coeffs ordered by (style_type, i, j,
    substyle, nsub), with style_type ordered as in STYLE_TYPES (pair,
    bond, angle, dihedral, improper) rather than alphabetically. substyle and
    nsub are included so multiple hybrid/overlay entries for the same
    pair (including repeated-name ones like 'lj/cut 1' vs 'lj/cut 2') sort
    deterministically."""
    return (
        STYLE_TYPES.index(coeff.style_type),
        coeff.substyle or "",
        coeff.nsub if coeff.nsub is not None else 0,
        coeff.i,
        coeff.j if coeff.j is not None else -1,
    )


def _expand_range(token, max_type):
    """
    Expand a single LAMMPS wildcard type token into a list of concrete ints.
    Supported forms: '*', 'n*', '*n', 'n*m', or a plain integer 'n'.
    `max_type` is used as the upper bound for an open-ended '*' or 'n*'.
    """
    if "*" not in token:
        return [int(token)]
    if token == "*":
        lo, hi = 1, max_type
    elif token.startswith("*"):
        lo, hi = 1, int(token[1:])
    elif token.endswith("*"):
        lo, hi = int(token[:-1]), max_type
    else:
        lo_s, hi_s = token.split("*")
        lo, hi = int(lo_s), int(hi_s)
    if hi < lo:
        return []
    return list(range(lo, hi + 1))


def _scan_max_types(filename):
    max_types = {st: 0 for st in STYLE_TYPES}

    for clean in _iter_input_lines(filename):
        parts = clean.split()
        keyword = parts[0]

        if (
            not keyword.endswith("_coeff")
            or keyword[: -len("_coeff")] not in STYLE_TYPES
        ):
            continue

        style_type = keyword[: -len("_coeff")]
        id_tokens = parts[1:3] if style_type == "pair" else parts[1:2]

        for tok in id_tokens:
            for num_tok in tok.split("*"):
                if num_tok.isdigit():
                    max_types[style_type] = max(
                        max_types[style_type],
                        int(num_tok),
                    )

    return max_types

class Style:
    """One *_style category, e.g. all of 'pair_style ...'."""

    def __init__(self, style_type, substyles=None, hybrid=False, scale_factors=None):
        self.style_type = style_type
        self.substyles = substyles or []
        self.hybrid = hybrid

        if scale_factors is None:
            self.scale_factors = [None] * len(self.substyles)
        else:
            if len(scale_factors) != len(self.substyles):
                raise ValueError(
                    "scale_factors must have the same length as substyles"
                )
            self.scale_factors = list(scale_factors)

    @property
    def names(self):
        """Bare substyle names (first token of each substyle string)."""
        return {s.split()[0] for s in self.substyles}

    @property
    def stripped_names(self):
        """Substyle names with accelerator suffixes (/kk, /omp) removed."""
        return {_strip_accelerator_suffix(n) for n in self.names}
    
    @property
    def nsubs(self):
        """1-based occurrence count per bare name, in order of appearance."""
        counts, out = {}, []
        for s in self.substyles:
            name = s.split()[0]
            counts[name] = counts.get(name, 0) + 1
            out.append(counts[name])
        return out

    def find(self, name, nsub):
        """Index in self.substyles of the nsub-th occurrence of `name`."""
        for i, (s, n) in enumerate(zip(self.substyles, self.nsubs)):
            if s.split()[0] == name and n == nsub:
                return i
        raise KeyError((name, nsub))


    def add_substyle(self, substyle_text, scale_factor=None):
        self.substyles.append(substyle_text)
        self.scale_factors.append(scale_factor)
        if len(self.substyles) > 1 and self.hybrid is False:
            self.hybrid = "hybrid/overlay"

    def substyle_name_count(self, name):
        """How many entries in self.substyles share this bare name -- >1 means
        pair_coeff/etc. lines for it must disambiguate with a nsub."""
        return sum(1 for s in self.substyles if s.split()[0] == name)

    def build_style_line(self):
        """Return the text of the *_style line, without the leading keyword."""
        if not self.substyles:
            return "none"

        has_scaling = any(factor is not None for factor in self.scale_factors)

        if has_scaling:
            factors = [
                "1" if factor is None else str(factor)
                for factor in self.scale_factors
            ]
            entries = [
                f"{factor} {substyle}"
                for factor, substyle in zip(factors, self.substyles)
            ]
            hybrid = "hybrid/scaled"
        elif self.hybrid is False:
            return " ".join(self.substyles)
        else:
            entries = self.substyles
            hybrid = self.hybrid

        prefix = f"{hybrid} "
        indent = " " * (
            len(f"{self.style_type}_style ") + len(prefix)
        )

        return prefix + f" &\n{indent}".join(entries)


    def write_style_line(self, outfile):
        outfile.write(f"{self.style_type}_style {self.build_style_line()}\n")

    def get_substyle_text(self, name, nsub=None):
        """Return the raw substyle text ('name arg1 arg2 ...') as declared on
        the *_style line, for the given bare substyle name. If the name
        appears more than once, nsub (1-based) picks which instance."""
        matches = [s for s in self.substyles if s.split()[0] == name]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        if nsub is None:
            raise ValueError(
                f"Substyle {name!r} appears {len(matches)} times in "
                f"{self.style_type}_style; nsub required to disambiguate"
            )
        return matches[nsub - 1]

    def __repr__(self):
        return f"Style({self.style_type!r}, hybrid={self.hybrid!r}, substyles={self.substyles!r})"


# --------------------------------------------------------------------------
# Coeff
# --------------------------------------------------------------------------

@dataclass
class Coeff:
    """One *_coeff line. For 'pair', both i and j are set; otherwise j is None.

    nsub disambiguates a substyle that's listed more than once in a
    hybrid/overlay declaration (e.g. 'pair_style hybrid/overlay lj/cut 1.0
    lj/cut 2.5' -- pair_coeff lines then refer to 'lj/cut 1' vs 'lj/cut 2').
    None means the substyle appears only once, so no index is needed/given.
    """

    style_type: str            # 'pair', 'bond', 'angle', 'dihedral', 'improper'
    i: int
    raw_substyle: str
    substyle: str              # bare substyle name, e.g. "lj/cut"
    args: list = field(default_factory=list)   # remaining numeric/string args
    j: int = None              # only used for 'pair'
    nsub: int = None   # 1-based instance number, only when the substyle repeats

    def build_coeff_line(self, hybrid):
        """Return the text of the *_coeff line, without the leading keyword."""
        if self.style_type == "pair":
            ids = [str(self.i), str(self.j)]
        else:
            ids = [str(self.i)]
        substyle_part = []
        if hybrid:
            substyle_part.append(self.substyle)
            if self.nsub is not None:
                substyle_part.append(str(self.nsub))
        return " ".join(ids + substyle_part + [str(a) for a in self.args])

    def write_coeff_line(self, outfile, hybrid):
        outfile.write(f"{self.style_type}_coeff {self.build_coeff_line(hybrid)}\n")

    def __repr__(self):
        idx_part = f" #{self.nsub}" if self.nsub is not None else ""
        if self.style_type == "pair":
            return f"Coeff(pair {self.i} {self.j}, {self.substyle}{idx_part}, {self.args})"
        return f"Coeff({self.style_type} {self.i}, {self.substyle}{idx_part}, {self.args})"


# --------------------------------------------------------------------------
# Forcefield
# --------------------------------------------------------------------------

class Forcefield:
    """A full set of styles + coefficients, with read/merge/reduce/clone/write."""

    def __init__(self, style_file=None, param_file=None, data_files=None, atom_style="full"):
        self.atom_style = atom_style
        self.styles = []      # list[Style]
        self.coeffs = []      # list[Coeff], always kept sorted (style_type, i, j)

        # Read datafile to get max type, in case of ONLY * * parameters
        if hasattr(data_files, "used_types"):
            data_files = [data_files]
            # data_files = [LammpsData(df, atom_style) for df in data_files if isinstance(df, str)]
        
        max_type = None
        if data_files:
            max_type = 1
            for df in data_files:
                used = df.used_types()
                max_type = max(max(used['pair']), max_type)
        
        if style_file:
            self.read_style_file(style_file)
        if param_file:
            self.read_parameter_file(param_file, max_type)

    # -- small lookup helpers ------------------------------------------------

    def find_accelerator(self, style_types=None):
        """
        Find every substyle using an accelerator suffix (/kk, /omp).

        Checks both the *_style declarations (Style.substyles) and the
        *_coeff raw_substyle text, since either source could carry the
        suffix independently of the other (e.g. a parameter file read
        without its matching style file).

        Only one accelerator may be in use across the whole forcefield --
        LAMMPS doesn't support mixing /kk and /omp substyles in a single
        run -- so this raises if both are found, even in different
        style_types.

        :param style_types: which style categories to search. Defaults to
            STYLE_TYPES.
        :return: {style_type: [(bare_name, accelerator), ...]}, sorted and
            deduplicated, where accelerator is 'kk' or 'omp' (no leading
            slash). Only style_types with at least one match are included.
        :rtype: dict
        :raises ValueError: if more than one accelerator type is found
            across the forcefield.
        """
        style_types = style_types or STYLE_TYPES
        accelerators = ("kk", "omp")

        found = {}
        seen_accelerators = set()

        for st in style_types:
            matches = set()

            style = self._get_style(st, create=False)
            if style is not None:
                for substyle in style.substyles:
                    name = substyle.split()[0]
                    for accel in accelerators:
                        if name.endswith(f"/{accel}"):
                            matches.add((name, accel))
                            seen_accelerators.add(accel)

            for coeff in self._iter_coeffs(st):
                if not coeff.raw_substyle:
                    continue
                name = coeff.raw_substyle.split()[0]
                for accel in accelerators:
                    if name.endswith(f"/{accel}"):
                        matches.add((name, accel))
                        seen_accelerators.add(accel)

            if matches:
                found[st] = sorted(matches)

        if len(seen_accelerators) > 1:
            raise ValueError(
                f"Multiple accelerators found in forcefield: "
                f"{sorted(seen_accelerators)}. Only one accelerator "
                f"(/kk or /omp) may be used at a time."
            )
        elif len(seen_accelerators) == 1:
            return list(seen_accelerators)[0]
        else:
            return None

    def _get_style(self, style_type, create=True):
        for style in self.styles:
            if style.style_type == style_type:
                return style
        if not create:
            return None
        style = Style(style_type)
        # Insert keeping self.styles ordered as in STYLE_TYPES (pair, bond,
        # angle, dihedral, improper), so output files follow that order too.
        target_idx = STYLE_TYPES.index(style_type)
        idx = 0
        while idx < len(self.styles) and STYLE_TYPES.index(self.styles[idx].style_type) < target_idx:
            idx += 1
        self.styles.insert(idx, style)
        return style

    def _iter_coeffs(self, style_type):
        return (c for c in self.coeffs if c.style_type == style_type)

    def _find_coeff(self, style_type, i, j=None, substyle=None, nsub=None, match_substyle=False):
        for c in self.coeffs:
            if c.style_type == style_type and c.i == i and c.j == j:
                if match_substyle and (c.substyle != substyle or c.nsub != nsub):
                    continue
                return c
        return None

    def _insert_coeff(self, coeff, replace=False):
        """Insert a Coeff, keeping self.coeffs sorted by (style_type, i, j, substyle, nsub).

        For 'pair', uniqueness is keyed on (i, j, substyle, nsub)
        rather than just (i, j), since hybrid/overlay pair styles legitimately
        stack several substyles (e.g. lj/cut *and* coul/wolf) -- or even the
        same substyle listed twice with different nsub (e.g. 'lj/cut
        1' and 'lj/cut 2') -- on the same type pair. Other categories never
        overlay, so they stay keyed on (i) alone.
        """
        if coeff.style_type == "pair":
            existing = self._find_coeff(
                coeff.style_type, coeff.i, coeff.j,
                substyle=coeff.substyle, nsub=coeff.nsub, match_substyle=True,
            )
        else:
            existing = self._find_coeff(coeff.style_type, coeff.i, coeff.j)

        if existing is not None:
            if coeff == existing:
                return  # identical entry already present -- nothing to do, not an error
            if not replace:
                if coeff.style_type == "pair":
                    idx_part = f" #{coeff.nsub}" if coeff.nsub is not None else ""
                    raise ValueError(
                        f"Duplicate, non-identical pair coeff for type ({coeff.i}, {coeff.j}) "
                        f"with substyle {coeff.substyle!r}{idx_part}."
                    )
                raise ValueError(f"Duplicate, non-identical {coeff.style_type} coeff for type {coeff.i}.")
            self.coeffs.remove(existing)

        key = _coeff_sort_key(coeff)
        idx = 0
        while idx < len(self.coeffs) and _coeff_sort_key(self.coeffs[idx]) < key:
            idx += 1
        self.coeffs.insert(idx, coeff)

    def add_coeff(self, style_type, i, substyle, args, j=None, nsub=None):
        style = self._get_style(style_type, create=False)
        raw_substyle = style.get_substyle_text(substyle, nsub) if style and substyle else None
        coeff = Coeff(
            style_type=style_type, i=i, j=j,
            raw_substyle=raw_substyle, substyle=substyle,
            args=list(args), nsub=nsub,
        )
        self._insert_coeff(coeff, replace=False)

    # -- reading --------------------------------------------------------------

    def read_style_file(self, filename):
        """Parse a LAMMPS style file (e.g. in.init) and add *_style lines."""
        
        for clean in _iter_input_lines(filename):
            parts = clean.split()
            if len(parts) < 2:
                continue
            keyword = parts[0]
            if not keyword.endswith("_style") or keyword[: -len("_style")] not in STYLE_TYPES:
                continue
            style_type = keyword[: -len("_style")]
            value = " ".join(parts[1:])

            style = self._get_style(style_type)
            if "hybrid" in parts[1]:
                style.hybrid = parts[1]
                substyles, scale_factors = _split_hybrid_style(value)

                for substyle, scale_factor in zip(substyles, scale_factors):
                    style.add_substyle(substyle, scale_factor)
            else:
                style.add_substyle(value)

        return self

    def read_parameter_file(self, filename, max_type=None):
        """Parse a LAMMPS parameter file (e.g. in.settings) and add *_coeff lines.

        Wildcard type tokens ('*', 'n*', '*n', 'n*m') are expanded into
        concrete, individual type entries at parse time (using the highest
        type number seen anywhere in this file as the bound for an
        open-ended '*'). A wildcard-expanded entry only fills in a type (or
        type pair) that doesn't already have an explicit entry -- it never
        overrides one. This keeps every stored Coeff fully explicit, so
        later merging/reducing/cloning never has to reason about a live
        wildcard whose meaning could shift as types are added or renumbered.
        """
        max_types = _scan_max_types(filename)
        if not max_types and max_type:
            max_types['pair'] = max_type


        for clean in _iter_input_lines(filename):
            parts = clean.split()
            keyword = parts[0]
            if not keyword.endswith("_coeff") or keyword[: -len("_coeff")] not in STYLE_TYPES:
                continue
            style_type = keyword[: -len("_coeff")]
            style = self._get_style(style_type, create=False) or Style(style_type)
            is_hybrid = len(style.substyles) > 1

            if style_type == "pair":
                i_tok, j_tok = parts[1], parts[2]
                wildcard = "*" in i_tok or "*" in j_tok
                rest = parts[3:]
                if wildcard:
                    i_range = _expand_range(i_tok, max_types["pair"])
                    j_range = _expand_range(j_tok, max_types["pair"])
                    targets = sorted({tuple(sorted((a, b))) for a in i_range for b in j_range})
                else:
                    targets = [tuple(sorted((int(i_tok), int(j_tok))))]
            else:
                i_tok = parts[1]
                wildcard = "*" in i_tok
                rest = parts[2:]
                targets = (
                    [(t,) for t in _expand_range(i_tok, max_types[style_type])]
                    if wildcard
                    else [(int(i_tok),)]
                )

            if wildcard and not targets:
                print(f"WARNING: could not resolve wildcard '{clean}' (unknown type range); skipped.")
                continue

            # Find and pull out the substyle token, if this category is hybrid.
            substyle = style.substyles[0].split()[0] if style.substyles else None
            args = rest
            for idx, tok in enumerate(rest):
                stripped = _strip_accelerator_suffix(tok)
                if tok in style.names or tok in style.stripped_names or stripped in style.stripped_names:
                    substyle = tok
                    args = rest[:idx] + rest[idx + 1:]
                    break

            if is_hybrid and substyle is None:
                raise ValueError(f"{filename}: hybrid {style_type}_style but no substyle in: {clean}")

            # If this substyle name is listed more than once in the style
            # declaration, LAMMPS requires the next token to be a 1-based
            # index picking which instance this line refers to (e.g.
            # 'lj/cut 1' vs 'lj/cut 2'). Consume it if present.
            nsub = None
            if substyle is not None and style.substyle_name_count(substyle) > 1:
                if not args or _try_int(args[0]) is None:
                    raise ValueError(
                        f"{filename}: substyle {substyle!r} appears more than once in "
                        f"{style_type}_style but no disambiguating index found in: {clean}"
                    )
                nsub = int(args[0])
                args = args[1:]

            for target in targets:
                i = target[0]
                j = target[1] if style_type == "pair" else None
                if wildcard:
                    if style_type == "pair":
                        already_set = self._find_coeff(
                            style_type, i, j, substyle=substyle, nsub=nsub, match_substyle=True,
                        )
                    else:
                        already_set = self._find_coeff(style_type, i, j)
                    if already_set is not None:
                        continue  # wildcard fills gaps only; never duplicates an existing (substyle-matched) entry
                raw_substyle = style.get_substyle_text(substyle, nsub) if substyle is not None else None
                coeff = Coeff(
                    style_type=style_type, i=i, j=j,
                    raw_substyle=raw_substyle, substyle=substyle,
                    args=list(args), nsub=nsub,
                )
                self._insert_coeff(coeff, replace=wildcard)
        
        return self

    # -- reducing -----------------------------------------------------------------

    def reduce_forcefield(self, data_files, atom_style):
        """
        Keep only the types actually used in `data_files` (str or list[str]),
        and renumber the surviving types to 1..N per category.
        Returns the old_type -> new_type mapping, per category.
        """
        if isinstance(data_files, str):
            data_files = [LammpsData(data_files)]

        used = {st: set() for st in STYLE_TYPES}
        for datafile in data_files:
            file_used = datafile.used_types()
            for st in STYLE_TYPES:
                used[st] |= file_used[st]

        type_mapping = {
            st: {old: new for new, old in enumerate(sorted(used[st]), start=1)}
            for st in STYLE_TYPES
        }

        new_coeffs = []
        for coeff in self.coeffs:
            st = coeff.style_type
            if coeff.i not in used[st] or (st == "pair" and coeff.j not in used[st]):
                continue
            new_i = type_mapping[st][coeff.i]
            if st == "pair":
                new_i, new_j = sorted((new_i, type_mapping[st][coeff.j]))
            else:
                new_j = None
            new_coeffs.append(Coeff(
                style_type=st, i=new_i, raw_substyle=coeff.raw_substyle,
                substyle=coeff.substyle, args=list(coeff.args),
                j=new_j, nsub=coeff.nsub,
            ))
        new_coeffs.sort(key=_coeff_sort_key)
        self.coeffs = new_coeffs

        # Drop now-unused substyles from each Style.
        for style in self.styles:
            used_substyle_names = {c.substyle for c in self._iter_coeffs(style.style_type) if c.substyle}
            kept = [
                (substyle, scale_factor)
                for substyle, scale_factor
                in zip(style.substyles, style.scale_factors)
                if substyle.split()[0] in used_substyle_names
            ]
            style.substyles = [s for s, _ in kept]
            style.scale_factors = [f for _, f in kept]

            if len(style.substyles) < 2:
                style.hybrid = False
            elif style.hybrid is False:
                if any(factor is not None for factor in style.scale_factors):
                    style.hybrid = "hybrid/scaled"
                else:
                    style.hybrid = "hybrid/overlay"


        return type_mapping

    # -- cloning --------------------------------------------------------------

    def clone_types(self, mapping, style_types=None):
        """
        Duplicate coeff entries for one or more atom types under caller-chosen
        new type ids.

        :param mapping: dict of {new_type: old_type}. Each new_type becomes a
            full copy of old_type's coeffs, across every style_type touched.
            Multiple new_types may map to the same old_type (several clones
            of one original in a single call).
        :type mapping: dict[int, int]
        :param style_types: which style categories to clone. Defaults to
            STYLE_TYPES.

        For "pair", cross terms are generated between every clone and every
        other type -- including between two different new_types cloned from
        *different* old_types in the same call (e.g. mapping={9: 1, 10: 2}
        yields a correct 9-10 term derived from the existing 1-2 term) --
        all in a single pass, with no dependence on dict iteration order.
        For non-pair (single-index) categories, each new_type just gets a
        copy of old_type's one coeff row.

        Raises ValueError if any new_type in `mapping` collides with an
        existing type id in a touched category. Silently skips a (new_type,
        style_type) pair if old_type has no coeff for that style_type,
        matching clone_type's prior behavior.

        :returns: {style_type: {new_type: old_type}} for clones actually made.
        :rtype: dict
        """
        style_types = style_types or STYLE_TYPES

        children = {}
        for new_t, old_t in mapping.items():
            children.setdefault(old_t, []).append(new_t)

        made = {}

        for st in style_types:
            existing_ids = {c.i for c in self._iter_coeffs(st)} | {
                c.j for c in self._iter_coeffs(st) if c.j is not None
            }
            collisions = set(mapping) & existing_ids
            if collisions:
                raise ValueError(f"{st}: new type id(s) {sorted(collisions)} already in use")

            if st == "pair":
                old_rows = [c for c in self._iter_coeffs(st) if c.i in children or c.j in children]
                if not old_rows:
                    continue
                for c in old_rows:
                    i_variants = [c.i] + children.get(c.i, [])
                    j_variants = [c.j] + children.get(c.j, [])
                    for i in i_variants:
                        for j in j_variants:
                            ni, nj = sorted((i, j))
                            self._insert_coeff(
                                Coeff(
                                    style_type=st, i=ni, raw_substyle=c.raw_substyle,
                                    substyle=c.substyle, args=list(c.args),
                                    j=nj, nsub=c.nsub,
                                ),
                                replace=True,
                            )
                made[st] = {n: o for n, o in mapping.items()}
            else:
                st_made = {}
                for new_t, old_t in mapping.items():
                    old_coeff = self._find_coeff(st, old_t)
                    if old_coeff is None:
                        continue
                    self._insert_coeff(
                        Coeff(st, new_t, old_coeff.substyle, list(old_coeff.args), nsub=old_coeff.nsub),
                        replace=True,
                    )
                    st_made[new_t] = old_t
                if st_made:
                    made[st] = st_made

        return made

    # -- writing ------------------------------------------------------------

    def write_style_file(self, filename):
        with open(filename, "w") as f:
            for style in self.styles:
                if style.substyles:
                    style.write_style_line(f)

    def write_parameter_file(self, filename):
        with open(filename, "w") as f:
            for style in self.styles:
                hybrid = style.hybrid is not False
                coeffs = list(self._iter_coeffs(style.style_type))  # already sorted
                if not coeffs:
                    continue
                for coeff in coeffs:
                    coeff.write_coeff_line(f, hybrid)
                f.write("\n")

    def add_substyle(self, style_type, substyle, scale_factor=None):
        style = self._get_style(style_type)
        style.add_substyle(substyle, scale_factor)
        basestyle = substyle.strip().split()[0]
        nrep = style.substyle_name_count(basestyle)
        if nrep == 2:
            for c in self._iter_coeffs(style_type):
                if c.substyle == basestyle and c.nsub is None:
                    c.nsub = 1
        if nrep == 1:
            return None
        else:
            return nrep

    def __repr__(self):
        counts = {st: len(list(self._iter_coeffs(st))) for st in STYLE_TYPES}
        return f"Forcefield(styles={[s.style_type for s in self.styles]}, coeff_counts={counts})"

