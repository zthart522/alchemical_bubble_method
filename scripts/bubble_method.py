from lammps_data import LammpsData
from ff_parsing import Forcefield, _iter_input_lines
from automate_utils import Scheduler, Checkpointer, fill_template
import os, shutil, json, copy
from time import sleep
import argparse
import random
import numpy as np

# Helper functions to act on Forcefield to prepare scaled interactions
# (i.e. need to determine solute types, determine pair_styles being used by solvent-solute,
# make copies of those, adjust nsub for solvent-solute interactions,
# and alter pair_style line to use hybrid/scaled w1 pair1 args1 w2 pair2 args2 ...)

STAGE_OUTPUTS = {
    "min": ["minimized.data"],
    "equil": ["equilibrated.data"],
}

def find_solute_types(datafile, attr, target_value):
    solute_types = set()
    for atom in datafile.Atoms:
        if getattr(atom, attr) == target_value:
            solute_types.add(atom.type)
    return list(sorted(solute_types))

def duplicate_cross_pairstyles(forcefield, solute_types, scale_factor='v_lambda'):
    style = forcefield._get_style('pair')
    coeffs = list(forcefield._iter_coeffs('pair'))
    keys = {id(c): (c.substyle, c.nsub or 1) for c in coeffs}   # snapshot, None -> 1
    duplicate_idx = {}
    dups = []

    for c in coeffs:
        if (c.i in solute_types) != (c.j in solute_types):
            continue                                   # cross pairs: original only
        key = keys[id(c)]
        if key not in duplicate_idx:
            idx = forcefield.add_substyle('pair', c.raw_substyle, scale_factor=scale_factor)
            if idx is None:
                raise ValueError(f"{c.substyle} not detected as repeated")
            orig = style.find(key[0], key[1])
            style.scale_factors[orig] = f'$(1.0-{scale_factor})'
            duplicate_idx[key] = idx

        d = copy.deepcopy(c)
        d.nsub = duplicate_idx[key]
        dups.append(d)

    for d in dups:
        forcefield._insert_coeff(d)

    return duplicate_idx

def add_bubble_potential(forcefield, solute_types, pair_style, args, scale_factor='v_lambda'):
    # Example usage: pair_style = 'soft 3.0'
    # args: f'10.0' (where 10.0 is the value of A for the soft potential)
    soft_style_idx = forcefield.add_substyle('pair', pair_style, scale_factor=scale_factor)
    atom_types = [c.i for c in forcefield._iter_coeffs('pair')]
    for i in atom_types:
        for j in atom_types:
            cross = (i in solute_types) != (j in solute_types)
            if i > j or not cross:
                continue

            forcefield.add_coeff(
                style_type='pair',
                i=min(i, j), 
                substyle=pair_style.strip().split()[0],
                args=args.strip().split(),
                j=max(i, j),
                nsub=soft_style_idx
            )

def parse_kspace_style(lammps_input):
    has_kspace = False
    with open(lammps_input) as f:
        for line in f:
            parts = line.strip().split()
            if parts and len(parts) >= 2:
                if parts[0] == 'kspace_style' and parts[1] != 'none':
                    has_kspace = True
                    break
    return has_kspace

def build_compute_dUdL_lines(lammps_filled_input, positive_match, negative_match=None):
    
    if negative_match is not None and positive_match == negative_match:
        raise ValueError("positive_match and negative_match must differ")

    # Find pair_style line
    pair_parts = None
    for clean in _iter_input_lines(lammps_filled_input):
        parts = clean.split()
        if parts and parts[0] == 'pair_style':
            pair_parts = parts
            break
    if pair_parts is None:
        raise ValueError("No pair_style line found")

    # Count instances of each token, both total and rolling
    nsub = []
    nsub_total = {}
    for part in pair_parts:
        nsub_total[part] = nsub_total.get(part, 0) + 1
        nsub.append(nsub_total[part])

    # Collect (substyle, sign) in order of appearance
    signed_styles = []
    for i, part in enumerate(pair_parts[:-1]):
        if part == positive_match:
            sign = '+'
        elif negative_match is not None and part == negative_match:
            sign = '-'
        else:
            continue
        substyle = pair_parts[i + 1]
        if nsub_total[substyle] > 1:
            substyle += f" {nsub[i + 1]}"
        signed_styles.append((substyle, sign))

    if not signed_styles:
        raise ValueError("No sub-styles matched positive_match or negative_match")

    # Build the lines
    lines = ""
    expr = ""
    for i, (style, sign) in enumerate(signed_styles):
        lines += f"compute\t\t\tps_{i} all pair {style}\n"
        if i == 0:
            expr = f"-c_ps_{i}" if sign == '-' else f"c_ps_{i}"
        else:
            expr += f"{sign}c_ps_{i}"
    lines += f"variable\t\tdUdL equal {expr}"

    return lines

def prepare_simulation(template, template_fill, directory, input_datafile, paramsfile, stylefile):

    # Make copy of templates
    fill_copy = template_fill.copy()

    # Make directory, if it doesn't exist, and copy files
    os.makedirs(directory, exist_ok=True)
    shutil.copy(input_datafile, directory)
    shutil.copy(stylefile, directory)
    shutil.copy(paramsfile, directory)

    # Add filenames to template
    fill_copy["data_infile"] = os.path.basename(input_datafile)
    fill_copy["forcefield_styles"] = os.path.basename(stylefile)
    fill_copy["forcefield_params"] = os.path.basename(paramsfile)
    
    # Fill templates
    fill_template(template, fill_copy, f'{directory}/lammps.in')

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return json.load(f)


def get_template(cfg, name):
    """Template for a stage. Legs fall back to a shared 'bubble' template."""
    templates = cfg["input_templates"]
    if name in templates:
        return templates[name]
    if "bubble" in templates:
        return templates["bubble"]
    raise KeyError(
        f"No input template for '{name}' and no 'bubble' fallback in input_templates"
    )


def make_job_fill(cfg, stage, lmp_exe, flags):
    """Build slurm fill dict: global job_details + scheduler + per-stage overrides."""
    fill = {k: v for k, v in cfg["job_details"].items() if v not in ("", None)}
    fill.setdefault("nodes", 1)
    fill["cmd"] = f'{lmp_exe} {flags} -in lammps.in | tee out.lammps'
    fill.update(cfg.get("ti_job_details", {}).get(stage, {}))
    return fill


def get_accelerator_flags(cfg, ff):
    acc = cfg.get("accelerator")
    if acc is None:
        acc = ff.find_accelerator()
    if acc == "kk":
        return "-k on -sf kk"
    if acc == "omp":
        return "-sf omp"
    return ""

def touch_files(directory, filenames):
    os.makedirs(directory, exist_ok=True)
    for name in filenames:
        path = os.path.join(directory, name)
        with open(path, "a"):
            os.utime(path, None)
        print(f"[dry-run] touched {path}")


def wait_for(scheduler, calc_ids, dry_run=True):
    if dry_run:
        return
    scheduler.block_until_complete(calc_ids)

def run_stage(cfg, stage, directory, lmp_exe, flags, scheduler, dry_run=True):
    job_fill = make_job_fill(cfg, stage, lmp_exe, flags)
    scheduler_template = cfg["input_templates"]["scheduler"]
    submit_name = os.path.basename(scheduler_template)
    fill_template(scheduler_template, job_fill, f"{directory}/{submit_name}")
    if dry_run:
        print(f"[dry-run] would submit {directory}/{submit_name}")
        touch_files(directory, STAGE_OUTPUTS.get(stage, []))
        return
    calc_id = scheduler.submit_job(f"{directory}/{submit_name}")
    scheduler.block_until_complete(calc_id)


def run_ti_leg(cfg, leg, lams, root, lmp_exe, flags, scheduler, dry_run=True):
    sched = cfg["scheduler"]
    job_fill = make_job_fill(cfg, leg, lmp_exe, flags)
    scheduler_template = cfg["input_templates"]["scheduler"]
    submit_name = os.path.basename(scheduler_template)
    calc_ids = []
    for lam in lams:
        lam_dir = f"{root}/{leg}/lambda_{lam:.5f}"
        fill_template(scheduler_template, job_fill, f"{lam_dir}/{submit_name}")

        if dry_run:
            print(f"[dry-run] would submit {lam_dir}/{submit_name}")
            touch_files(lam_dir, [f"{leg}_{lam:.5f}.data"])
            continue

        calc_id = scheduler.submit_job(
            submit_file=f"{lam_dir}/{submit_name}"
        )
        calc_ids.append(calc_id)
        sleep(5)
    return calc_ids

def parse_group_cmd(str):
    parts = str.strip().split()
    if parts[0] == "molecule":
        return "molecule_id", int(parts[1])
    elif parts[0] == "id":
        return "id", int(parts[1])
    else:
        raise ValueError(
            f"group_cmd expects 'molecule <id>', 'type <id>', or "
            f"'id <id>', got: {str!r}"
        )

LAMBDA_MAP = {
    "ti_cross":"lambda_cross",
    "bubble_on": "lambda_bubble",
    "bubble_off": "lambda_bubble"
}

# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def main(config_path, root=".", dry_run=True):
    
    # ---------------- Read config ----------------
    cfg = load_config(config_path)
    sysp = cfg["system_params"]
    lmp_exe = cfg["executables"]["lammps"]
    scheduler = Scheduler(**cfg["scheduler"])

    os.makedirs(f"{root}/pack", exist_ok=True)
    os.makedirs(f"{root}/ForceField", exist_ok=True)

    # ---------------- Build system ----------------
    if sysp["system_mode"] == "pack":
        
        # Solute first (molecule 1), then solvent(s)
        passed = [{"structure": sysp["solute"]["structure"], "number": 1}]
        passed += sysp["solvent"]

        molecules = []
        for mol in passed:
            molecules.append({
                "data": LammpsData(mol["structure"], cfg["sim_params"]["atom_style"]),
                "number": mol["number"],
            })

        # ---------------- Read forcefield ----------------
        ff = Forcefield(
            style_file=sysp["style_file"], 
            param_file=sysp["param_file"],
            data_files=[m["data"] for m in molecules],
            atom_style=cfg["sim_params"]["atom_style"]
        )

        # Reduce forcefield, using all types in all datafiles
        mapping = ff.reduce_forcefield([m["data"] for m in molecules], cfg["sim_params"]["atom_style"])
        for mol in molecules:
            mol["data"].apply_reduced_mapping(mapping)

        L = sysp["pack_boxlen"]
        system = LammpsData.pack(
            molecules=molecules,
            box=(0, L, 0, L, 0, L),
            packmol_exe=cfg["executables"]["packmol"],
            pack_dir=f"{root}/pack",
            tolerance=sysp["pack_tol"],  # adjust to your pack() signature
        )

        sol_group = 'molecule 1'
    else:
        
        # Read the already packed system & reduce the forcefield
        system = LammpsData(sysp.get("data_file"), cfg["sim_params"]["atom_style"])
        # ---------------- Read forcefield ----------------
        ff = Forcefield(
            style_file=sysp["style_file"], 
            param_file=sysp["param_file"],
            data_files=[system],
            atom_style=cfg["sim_params"]["atom_style"]
        )
        mapping = ff.reduce_forcefield([system], cfg["sim_params"]["atom_style"])
        system.apply_reduced_mapping(mapping)
        sol_group = cfg['system_params']['group_cmd']
        
    # Move group_cmd to sim_params
    cfg['sim_params']['group_cmd'] = sol_group
    attr, target = parse_group_cmd(sol_group)

    # Make sure solute is unique
    unique_attr_map = system.make_attr_unique(attr=attr, target_value=target)
    system.write_data(f"{root}/pack/packed.data")

    # For any new types required to make solute unique, clone relevant parameters
    ff.clone_types(unique_attr_map, style_types=["pair"])

    # Save original parameters before adding bubble/scaling
    ff.write_parameter_file(f"{root}/ForceField/params.lmp")
    ff.write_style_file(f"{root}/ForceField/styles.lmp")

    # ---------------- Modify forcefield for TI ----------------
    sol_types = find_solute_types(system, attr, target)
    duplicated_styles = duplicate_cross_pairstyles(
        ff, sol_types, scale_factor="v_lambda_cross"
    )

    bubble_pot = cfg.get("ti_params").get("bubble_potential")
    bubble_args = cfg.get("ti_params").get("bubble_param_args")
    add_bubble_potential(
        ff, sol_types, bubble_pot, bubble_args, scale_factor="v_lambda_bubble"
    )
    
    ff.write_parameter_file(f"{root}/ForceField/ti_params.lmp")
    ff.write_style_file(f"{root}/ForceField/ti_styles.lmp")

    # ---------------- Charge scaling needed? ----------------
    has_kspace = parse_kspace_style(cfg["input_templates"]["equil"])
    if has_kspace:
        raise ValueError("This pipeline should not be used with a kspace solver.")
    legs = ["bubble_on", "bubble_off", "ti_cross"]

    # ---------------- Template fill values ----------------
    template_fill = dict(cfg["sim_params"])
    template_fill["random_seed"] = (
        random.randint(1, 10**6) if cfg.get("random_seed_use") else 3454
    )
    template_fill["equil_frac"] = cfg["ti_params"]["equil_frac"]

    flags = get_accelerator_flags(cfg, ff)
    
    if sysp["system_mode"] == "pack":
        # ---------------- Minimization ----------------
        prepare_simulation(
            template=cfg["input_templates"]["min"],
            template_fill=template_fill,
            directory=f"{root}/min",
            input_datafile=f"{root}/pack/packed.data",
            stylefile=f"{root}/ForceField/styles.lmp",
            paramsfile=f"{root}/ForceField/params.lmp",
        )
        run_stage(cfg, "min", f"{root}/min", lmp_exe, flags, scheduler, dry_run=dry_run)
        min_datafile = f"{root}/min/minimized.data"
    else:
        min_datafile = f"{root}/pack/packed.data"

    # ---------------- Equilibration ----------------
    prepare_simulation(
        template=cfg["input_templates"]["equil"],
        template_fill=template_fill,
        directory=f"{root}/equil",
        input_datafile=min_datafile,
        stylefile=f"{root}/ForceField/styles.lmp",
        paramsfile=f"{root}/ForceField/params.lmp",
    )
    run_stage(cfg, "equil", f"{root}/equil", lmp_exe, flags, scheduler, dry_run=dry_run)

    # ---------------- Setup TI jobs (no running yet) ----------------
    # lambda_values = number of intervals
    lams = cfg["ti_params"]["lambda_values"]
    if isinstance(lams, int):
        lams = [float(x) for x in np.linspace(0.0, 1.0, lams)]
    elif not isinstance(lams, list):
        raise ValueError(f"lambda_values should be type list or int, got {lams}")

    for leg in legs:
        fill_copy = template_fill.copy()

        if leg == "bubble_on":
            fill_copy["lambda_cross"] = 0.0
        elif leg == "ti_cross":
            fill_copy["lambda_bubble"] = 1.0
        elif leg == "bubble_off":
            fill_copy["lambda_cross"] = 1.0
        else:
            raise ValueError(f"Unknown LEG: {leg}")
        
        fill_copy["dUdL_computation_block"] = build_compute_dUdL_lines(
            f"{root}/ForceField/ti_styles.lmp", f"v_{LAMBDA_MAP[leg]}",
            f"$(1.0-v_{LAMBDA_MAP[leg]})"
        )

        for lam in lams:
            fill_copy[LAMBDA_MAP[leg]] = lam

            lam_dir = f"{root}/{leg}/lambda_{lam:.5f}"
            os.makedirs(lam_dir, exist_ok=True)

            fill_copy["data_outfile"] = f"{leg}_{lam:.5f}.data"
            prepare_simulation(
                template=get_template(cfg, leg),
                template_fill=fill_copy,
                directory=lam_dir,
                input_datafile=f"{root}/equil/equilibrated.data",
                stylefile=f"{root}/ForceField/ti_styles.lmp",
                paramsfile=f"{root}/ForceField/ti_params.lmp",
            )

    # ---------------- Run simulations ----------------
    # Bubble on first
    calc_ids = run_ti_leg(cfg, "bubble_on", lams, root, lmp_exe, flags, scheduler, dry_run=dry_run)
    wait_for(scheduler, calc_ids, dry_run=dry_run)

    # Then any other transformations, concurrently
    calc_ids = []
    for leg in legs:
        if leg not in ("bubble_on", "bubble_off"):
            calc_ids.extend(run_ti_leg(cfg, leg, lams, root, lmp_exe, flags, scheduler, dry_run=dry_run))
    wait_for(scheduler, calc_ids, dry_run=dry_run)

    # Finally, turn the bubble back off
    calc_ids = run_ti_leg(cfg, "bubble_off", lams, root, lmp_exe, flags, scheduler, dry_run=dry_run)
    wait_for(scheduler, calc_ids, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--root", default=".", help="Output/working directory")
    parser.add_argument("--run", dest="dry_run", action="store_false",
                        help="Actually submit jobs. Default is a dry run: build "
                        "inputs and slurm scripts, touch expected outputs.")
    args = parser.parse_args()
    main(args.config, args.root, dry_run=args.dry_run)
