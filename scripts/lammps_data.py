from dataclasses import dataclass, field
import os
import subprocess
from random import randint


def read_xyz(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    natoms = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + natoms]:
        p = line.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def write_packmol_script(mol_specs, box, filetype="xyz", output_file="packed.xyz",
                         tolerance=2.0, script_path="packmol.inp", seed=None):
    seed = seed or randint(1, 10000)
    xlo, xhi, ylo, yhi, zlo, zhi = box

    with open(script_path, "w") as f:
        f.write(f"tolerance {tolerance}\n")
        f.write(f"filetype {filetype}\n")
        f.write(f"output {output_file}\n")
        f.write(f"seed {seed}\n\n")
        for spec in mol_specs:
            f.write(f"structure {spec['structure']}\n")
            f.write(f"  number {spec['number']}\n")
            f.write(f"  inside box {xlo} {ylo} {zlo} {xhi} {yhi} {zhi}\n")
            f.write("end structure\n\n")
    return script_path


def run_packmol(exe, script_path):
    with open(script_path) as stdin_file:
        result = subprocess.run(
            exe,
            stdin=stdin_file,
            cwd=os.path.dirname(script_path) or ".",
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Packmol failed (return code {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")
    return result

def _try_int(tok):
    try:
        return int(tok)
    except (TypeError, ValueError):
        return None


@dataclass
class Mass:
    type: int
    mass: float
    comment: str = ""


@dataclass
class Atom:
    id: int
    type: int
    molecule_id: int = None
    charge: float = None
    x: float = None
    y: float = None
    z: float = None
    nx: int = None
    ny: int = None
    nz: int = None
    comment: str = ""


@dataclass
class Bond:
    id: int
    type: int
    i: int
    j: int
    comment: str = ""


@dataclass
class Angle:
    id: int
    type: int
    i: int
    j: int
    k: int
    comment: str = ""


@dataclass
class Dihedral:
    id: int
    type: int
    i: int
    j: int
    k: int
    l: int
    comment: str = ""


@dataclass
class Improper:
    id: int
    type: int
    i: int
    j: int
    k: int
    l: int
    comment: str = ""


class LammpsData:

    SECTION_NAMES = (
        "Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"
    )

    IGNORED_SECTIONS = (
        "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", 
        "Dihedral Coeffs", "Improper Coeffs", "Velocities", 
        "Ellipsoids", "Lines", "Triangles", "Bodies"
    )

    ROW_CLASSES = {
        "Masses": Mass,
        "Atoms": Atom,
        "Bonds": Bond,
        "Angles": Angle,
        "Dihedrals": Dihedral,
        "Impropers": Improper,
    }

    TYPE_SECTIONS = {
        "Atoms": "pair",
        "Bonds": "bond",
        "Angles": "angle",
        "Dihedrals": "dihedral",
        "Impropers": "improper",
    }

    def __init__(self, filename=None, atom_style="full"):
        self.atom_style = atom_style
        self.header = []
        self.Masses = []
        self.Atoms = []
        self.Bonds = []
        self.Angles = []
        self.Dihedrals = []
        self.Impropers = []
        self.Pair_Coeffs = []
        self.type_counts = {"pair": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}

        if filename:
            self.read_data(filename)

    _TYPE_COUNT_KEYS = {
        "atom types": "pair",
        "bond types": "bond",
        "angle types": "angle",
        "dihedral types": "dihedral",
        "improper types": "improper",
    }

    def read_data(self, filename):
        self.header = []
        for name in self.SECTION_NAMES:
            setattr(self, name.replace(" ", "_"), [])
        self.type_counts = {"pair": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}

        section = None

        with open(filename) as f:
            for line in f:
                clean = line.split("#", 1)[0].strip()

                if clean in self.SECTION_NAMES:
                    section = clean
                    continue

                if clean in self.IGNORED_SECTIONS:
                    section = 'SKIP'
                
                if section == 'SKIP':
                    continue

                if section is None:
                    self.header.append(line)
                    tokens = clean.split()
                    if tokens:
                        key = " ".join(tokens[1:])
                        if key in self._TYPE_COUNT_KEYS:
                            self.type_counts[self._TYPE_COUNT_KEYS[key]] = int(tokens[0])
                elif clean:
                    getattr(self, section.replace(" ", "_")).append(
                        self._parse_line(section, line)
                    )

        return self

    def _parse_line(self, section, line):
        clean, _, comment = line.partition("#")
        parts = clean.split()
        comment = comment.strip()

        if section == "Masses":
            return Mass(int(parts[0]), float(parts[1]), comment)

        if section == "Atoms":
            style = self.atom_style.lower()
            if style == "full":
                nx, ny, nz = (None, None, None) if len(parts) == 7 else map(int, parts[7:10])
                return Atom(
                    id=int(parts[0]),
                    molecule_id=int(parts[1]),
                    type=int(parts[2]),
                    charge=float(parts[3]),
                    x=float(parts[4]),
                    y=float(parts[5]),
                    z=float(parts[6]),
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    comment=comment,
                )
            if style == "charge":
                nx, ny, nz = (None, None, None) if len(parts) == 6 else map(int, parts[7:10])
                return Atom(
                    id=int(parts[0]),
                    type=int(parts[1]),
                    charge=float(parts[2]),
                    x=float(parts[3]),
                    y=float(parts[4]),
                    z=float(parts[5]),
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    comment=comment,
                )
            if style == "atomic":
                nx, ny, nz = (None, None, None) if len(parts) == 5 else map(int, parts[7:10])
                return Atom(
                    id=int(parts[0]),
                    type=int(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    comment=comment,
                )
            raise ValueError(f"Unsupported atom style: {self.atom_style}")

        cls = self.ROW_CLASSES.get(section)
        if cls is None:
            return line

        n = len(cls.__dataclass_fields__) - 1
        return cls(*map(int, parts[:n]), comment=comment)

    def make_attr_unique(self, attr, target_value):
        target_atoms = [a for a in self.Atoms if getattr(a, attr) == target_value]
        if not target_atoms:
            raise ValueError(f"No atoms found with molecule_id == {target_value}")

        other_types = {a.type for a in self.Atoms if getattr(a, attr) != target_value}
        old_to_new = {}
        next_type = self.type_counts["pair"] + 1

        for a in target_atoms:
            if a.type in other_types and a.type not in old_to_new:
                old_to_new[a.type] = next_type
                next_type += 1

        new_to_old = {new_t: old_t for old_t, new_t in old_to_new.items()}

        for new_t, old_t in new_to_old.items():
            old_mass = next((m for m in self.Masses if m.type == old_t), None)
            if old_mass is not None:
                self.Masses.append(Mass(new_t, old_mass.mass, old_mass.comment))

        for new_t, old_t in new_to_old.items():
            for line in list(self.Pair_Coeffs):
                body_part, _, comment_part = line.partition("#")
                body = body_part.split()
                if not body or _try_int(body[0]) != old_t:
                    continue
                new_body = [str(new_t)] + body[1:]
                new_line = " ".join(new_body)
                comment = comment_part.strip()
                if comment:
                    new_line += f"  # {comment}"
                self.Pair_Coeffs.append(new_line)

        for a in target_atoms:
            if a.type in old_to_new:
                a.type = old_to_new[a.type]

        self.type_counts["pair"] = next_type - 1
        self._update_header()
        return new_to_old

    def apply_reduced_mapping(self, type_mapping):
        for section, style_type in self.TYPE_SECTIONS.items():
            mapping = type_mapping.get(style_type, {})
            for row in getattr(self, section):
                if row.type not in mapping:
                    continue
                row.type = mapping[row.type]

        mapping = type_mapping.get("pair", {})
        self.Masses = [
            Mass(mapping[m.type], m.mass, m.comment)
            for m in self.Masses
            if m.type in mapping
        ]

        # type_counts must reflect the FULL reduced type space the mapping
        # defines, not just what this file's rows happen to use.
        for style_type, mapping in type_mapping.items():
            if style_type in self.type_counts:
                self.type_counts[style_type] = max(mapping.values(), default=0)


        self._update_header()
        return self


    def used_types(self):
        return {
            "pair": {x.type for x in self.Atoms},
            "bond": {x.type for x in self.Bonds},
            "angle": {x.type for x in self.Angles},
            "dihedral": {x.type for x in self.Dihedrals},
            "improper": {x.type for x in self.Impropers},
        }

    def _count(self, section):
        return len(getattr(self, section))

    def _update_header(self, type_counts=None):
        counts = {
            "atoms": self._count("Atoms"),
            "bonds": self._count("Bonds"),
            "angles": self._count("Angles"),
            "dihedrals": self._count("Dihedrals"),
            "impropers": self._count("Impropers"),
            "atom types": self.type_counts["pair"],
            "bond types": self.type_counts["bond"],
            "angle types": self.type_counts["angle"],
            "dihedral types": self.type_counts["dihedral"],
            "improper types": self.type_counts["improper"],
        }

        if type_counts:
            counts.update(type_counts)

        new_header = []

        for line in self.header:
            parts = line.split()

            if len(parts) >= 3:
                key = " ".join(parts[1:3]).lower()
            elif len(parts) >= 2:
                key = parts[1].lower()
            else:
                key = None

            if key in counts:
                indent = line[:len(line) - len(line.lstrip())]
                new_header.append(
                    f"{indent}{counts[key]} {' '.join(parts[1:])}\n"
                )
            else:
                new_header.append(line)

        self.header = new_header

    def write_data(self, filename):
        sections = (
            ("Masses", self.Masses),
            ("Atoms", self.Atoms),
            ("Bonds", self.Bonds),
            ("Angles", self.Angles),
            ("Dihedrals", self.Dihedrals),
            ("Impropers", self.Impropers),
            ("Pair Coeffs", self.Pair_Coeffs),
        )

        with open(filename, "w") as f:
            f.writelines(self.header)

            for name, rows in sections:
                if not rows:
                    continue

                f.write(f"\n{name}\n\n")
                for row in rows:
                    f.write(self._format_row(name, row) + "\n")

        return filename

    def _format_row(self, section, row):
        if section == "Masses":
            values = [row.type, row.mass]
        elif section == "Atoms":
            style = self.atom_style.lower()
            if style == "full":
                values = [row.id, row.molecule_id, row.type, row.charge, row.x, row.y, row.z]
            elif style == "charge":
                values = [row.id, row.type, row.charge, row.x, row.y, row.z]
            elif style == "atomic":
                values = [row.id, row.type, row.x, row.y, row.z]
            else:
                raise ValueError(f"Unsupported atom style: {self.atom_style}")
            if row.nx is not None:
                values += [row.nx, row.ny, row.nz]
        elif section == "Bonds":
            values = [row.id, row.type, row.i, row.j]
        elif section == "Angles":
            values = [row.id, row.type, row.i, row.j, row.k]
        elif section in ("Dihedrals", "Impropers"):
            values = [row.id, row.type, row.i, row.j, row.k, row.l]
        else:
            return row.rstrip("\n")

        line = " ".join(map(str, values))
        return f"{line}  # {row.comment}" if row.comment else line

    # -- box --------------------------------------------------------------

    _BOX_KEYS = (("xlo", "xhi"), ("ylo", "yhi"), ("zlo", "zhi"))

    def get_box(self):
        bounds = {}
        for line in self.header:
            parts = line.split("#", 1)[0].split()
            if len(parts) == 4 and (parts[2], parts[3]) in self._BOX_KEYS:
                bounds[(parts[2], parts[3])] = (float(parts[0]), float(parts[1]))
        return tuple(v for key in self._BOX_KEYS for v in bounds.get(key, (-0.5, 0.5)))

    def set_box(self, box):
        xlo, xhi, ylo, yhi, zlo, zhi = box
        replacements = {
            ("xlo", "xhi"): f"{xlo:.6f} {xhi:.6f} xlo xhi\n",
            ("ylo", "yhi"): f"{ylo:.6f} {yhi:.6f} ylo yhi\n",
            ("zlo", "zhi"): f"{zlo:.6f} {zhi:.6f} zlo zhi\n",
        }
        found = set()
        new_header = []
        for line in self.header:
            parts = line.split("#", 1)[0].split()
            key = (parts[2], parts[3]) if len(parts) == 4 else None
            if key in replacements:
                new_header.append(replacements[key])
                found.add(key)
            else:
                new_header.append(line)
        for key in self._BOX_KEYS:
            if key not in found:
                new_header.append(replacements[key])
        self.header = new_header

    # -- packing ------------------------------------------------------------

    def write_xyz(self, filepath, comment="", element_map=None):
        with open(filepath, "w") as f:
            f.write(f"{len(self.Atoms)}\n")
            f.write(f"{comment}\n")
            for a in self.Atoms:
                label = element_map[a.type] if element_map else str(a.type)
                f.write(f"{label} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n")

    def types_match(self, other):
        return (
            self.type_counts == other.type_counts
            and sorted(self.Masses, key=lambda m: m.type) == sorted(other.Masses, key=lambda m: m.type)
            and sorted(self.Pair_Coeffs) == sorted(other.Pair_Coeffs)
        )

    @classmethod
    def combine(cls, molecules, atom_style="full", box=None):
        combined = cls(atom_style=atom_style)
        combined.header = [
            "LAMMPS data file\n", "\n",
            "0 atoms\n",
            "0 bonds\n",
            "0 angles\n",
            "0 dihedrals\n",
            "0 impropers\n",
            "\n",
            "0 atom types\n",
            "0 bond types\n",
            "0 angle types\n",
            "0 dihedral types\n",
            "0 improper types\n",
            "\n",
            "-0.5 0.5 xlo xhi\n", "-0.5 0.5 ylo yhi\n", "-0.5 0.5 zlo zhi\n",
        ]

        seen = []
        section_counters = {s: 0 for s in ("Bonds", "Angles", "Dihedrals", "Impropers")}
        atom_id = 0
        mol_id = 0

        for m in molecules:
            mol_data, n_copies = m["data"], m["number"]

            match = next((offs for d, offs in seen if d.types_match(mol_data)), None)
            if match is not None:
                offsets = match
            else:
                offsets = dict(combined.type_counts)  # {"pair": N, "bond": N, ...} before this mol is added
                for mass in mol_data.Masses:
                    combined.Masses.append(Mass(mass.type + offsets["pair"], mass.mass, mass.comment))
                for line in mol_data.Pair_Coeffs:
                    body, _, comment = line.partition("#")
                    tok = body.split()
                    new_line = " ".join([str(int(tok[0]) + offsets["pair"])] + tok[1:])
                    if comment.strip():
                        new_line += f"  # {comment.strip()}"
                    combined.Pair_Coeffs.append(new_line)
                for key in combined.type_counts:
                    combined.type_counts[key] += mol_data.type_counts[key]
                seen.append((mol_data, offsets))

            for _ in range(n_copies):
                mol_id += 1
                id_map = {}

                for a in mol_data.Atoms:
                    atom_id += 1
                    id_map[a.id] = atom_id
                    combined.Atoms.append(Atom(
                        id=atom_id, type=a.type + offsets["pair"], molecule_id=mol_id,
                        charge=a.charge, x=a.x, y=a.y, z=a.z, comment=a.comment,
                    ))

                for section in ("Bonds", "Angles", "Dihedrals", "Impropers"):
                    row_cls = cls.ROW_CLASSES[section]
                    offkey = cls.TYPE_SECTIONS[section]
                    id_fields = [f for f in row_cls.__dataclass_fields__ if f not in ("id", "type", "comment")]
                    for row in getattr(mol_data, section):
                        section_counters[section] += 1
                        kwargs = {f: id_map[getattr(row, f)] for f in id_fields}
                        getattr(combined, section).append(row_cls(
                            id=section_counters[section],
                            type=row.type + offsets[offkey],
                            comment=row.comment,
                            **kwargs,
                        ))

        if box is not None:
            combined.set_box(box)
        combined._update_header()
        return combined

    def substitute_coordinates(self, packed_atoms):
        if len(packed_atoms) != len(self.Atoms):
            raise ValueError(
                f"Atom count mismatch: {len(packed_atoms)} packed vs {len(self.Atoms)} in this system"
            )
        for atom, (_, x, y, z) in zip(self.Atoms, packed_atoms):
            atom.x, atom.y, atom.z = x, y, z

    @classmethod
    def pack(cls, molecules, box, tolerance=2.0, pack_dir="./",
              packmol_exe="packmol", atom_style="full", seed=None):
        os.makedirs(pack_dir, exist_ok=True)
        pack_dir_abs = os.path.abspath(pack_dir)

        mol_specs = []
        for idx, m in enumerate(molecules):
            xyz_name = f"molecule_{idx}.xyz"
            m["data"].write_xyz(os.path.join(pack_dir_abs, xyz_name))
            mol_specs.append({"structure": xyz_name, "number": m["number"]})

        output_filename = "system_packed.xyz"
        script_path = os.path.join(pack_dir, "packmol.inp")
        write_packmol_script(mol_specs, box, filetype="xyz",
                             output_file=output_filename, tolerance=tolerance,
                             script_path=script_path, seed=seed)
        run_packmol(packmol_exe, script_path)

        packed_atoms = read_xyz(os.path.join(pack_dir_abs, output_filename))

        expected = sum(m["number"] * len(m["data"].Atoms) for m in molecules)
        if expected != len(packed_atoms):
            raise ValueError(
                f"Atom count mismatch: expected {expected}, got {len(packed_atoms)} -- check molecule ordering"
            )

        combined = cls.combine(molecules, atom_style=atom_style, box=box)
        combined.substitute_coordinates(packed_atoms)
        return combined