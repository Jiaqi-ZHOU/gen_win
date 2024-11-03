#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec
from ase.units import Bohr, Ha, Ry
from lxml import etree
import xml.etree.ElementTree as ET
import json
import os
from shutil import copyfile, rmtree

# https://github.com/aiidateam/aiida-wannier90-workflows/tree/main/src/aiida_wannier90_workflows/utils/pseudo/data/semicore
jsonname = "/home/jzhou/input/aiida/PseudoDojo_0.4_PBE_SR_standard_upf_add_projwfc.json"


def get_atoms(filename):
    root = ET.parse(filename).getroot()
    # for child in root:
    #     print(child)
    input_ = root[2]
    atomic_structure = input_[2]
    atomic_positions = atomic_structure[0]
    atoms = []
    for i in range(len(atomic_positions)):
        atom = list(atomic_positions[i].attrib.values())[0]
        atoms.append(atom)
    return atoms


def get_num_wan(filename):
    with open(jsonname) as f:
        data = json.load(f)

    atoms = get_atoms(filename)
    natoms = len(atoms)

    num_wan = 0
    for i in range(natoms):
        pswfcs = data[atoms[i]]["pswfcs"]
        semicores = data[atoms[i]]["semicores"]
        wans = list(set(pswfcs) - set(semicores))
        nwans = len(wans)

        for j in range(nwans):
            orbit = wans[j][1]
            num_wan = num_wan + convert(orbit)

    return num_wan


def get_projection(filename):
    with open(jsonname) as f:
        data = json.load(f)

    atoms = get_atoms(filename)
    natoms = len(atoms)
    projections = []
    species = []
    for i in range(natoms):
        pswfcs = data[atoms[i]]["pswfcs"]
        semicores = data[atoms[i]]["semicores"]
        wans = list(set(pswfcs) - set(semicores))
        nwans = len(wans)
        atom = atoms[i]

        if atom not in species:
            species.append(atom)
            projections.append(atom)
            projections.append(":")

            for j in range(nwans):
                orbit = wans[j][1]
                projections.append(orbit)
                if j != nwans - 1:
                    projections.append(";")
            projections.append("\n")

    return projections


def get_nexbands(filename):
    with open(jsonname) as f:
        data = json.load(f)

    atoms = get_atoms(filename)
    natoms = len(atoms)

    nexbands = 0
    for i in range(natoms):
        semicores = data[atoms[i]]["semicores"]
        nsemicores = len(semicores)

        for j in range(nsemicores):
            semicore = semicores[j][1]
            # print(f"{atoms[i]=} {semicores=} {semicore=}")
            nexbands = nexbands + convert(semicore)
    # print(f"{nexbands=}")
    return nexbands


def convert(orbit):
    spinor = 2
    if orbit == "S":
        n = 1 * spinor
    elif orbit == "P":
        n = 3 * spinor
    elif orbit == "D":
        n = 5 * spinor
    else:
        n = 0
    return n


# This function is used to extract many children with the same name.
# In this case, this function will produce a list with len > 1, then I
# can play with it.
def get_xml_element(filename, parent, tag):
    tree = etree.parse(filename)
    # parent='band_structure'
    path = f"/qes:espresso/output/{parent}"
    if tag != "":
        path = path + f"/{tag}"
    elem = tree.xpath(
        path, namespaces={"qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0"}
    )
    return elem


def get_bands_array(filename, parent="band_structure", tag="ks_energies"):
    ex_bands = get_nexbands(filename)

    root = ET.parse(filename).getroot()
    output = root[3]
    band_structure = output[-1]
    nbnd = int(band_structure[3].text)

    nelec = int(float(band_structure[4].text))
    bands_array_list = list(get_xml_element(filename, parent, tag))

    nkpt = len(bands_array_list)
    kpt_array = np.zeros((nkpt, 3))
    bands_array = np.zeros((nkpt, nbnd))

    for i in range(nkpt):
        kpt_array[i, :] = [
            float(x) for x in bands_array_list[i].find("k_point").text.split()
        ]
        bands_array[i, :] = [
            float(x) for x in bands_array_list[i].find("eigenvalues").text.split()
        ]
    bands_array = bands_array * Ha

    VBM_band_index = nelec - 1
    VBM = np.max(bands_array[:, VBM_band_index])

    CBM_band_index = nelec
    CBM = np.min(bands_array[:, CBM_band_index])

    return ex_bands, VBM, CBM


def gen_kpath_from_DFT(bandsinfile):
    file = open(bandsinfile, "r")
    lines = file.readlines()
    for index, _ in enumerate(lines):
        line = _.split()
        if len(line) == 2 and line[0] == "K_POINTS":
            index_K_POINTS = index
            nkpt = int(lines[index_K_POINTS + 1])
    kpath = []
    for i in range(index_K_POINTS + 2, index_K_POINTS + 2 + nkpt):
        line_split = lines[i].split()

        # Remove 1.0000000 from bands.in
        kpath_split = ""
        for j in range(3):  # x, y, z
            space = " " * (20 - len(line_split[j + 1]))
            kpath_split = kpath_split + line_split[j] + space
        kpath.append(kpath_split + "\n")

    return kpath


def get_scf_fermi(scfxmlfile):
    root = ET.parse(scfxmlfile).getroot()
    input = root.find("input")
    nk1 = int(input.find("k_points_IBZ").find("monkhorst_pack").get("nk1"))
    nk2 = int(input.find("k_points_IBZ").find("monkhorst_pack").get("nk2"))
    nk3 = int(input.find("k_points_IBZ").find("monkhorst_pack").get("nk3"))
    nk = np.array([nk1, nk2, nk3])

    output = root.find("output")
    fermi = float(output.find("band_structure").find("fermi_energy").text) * Ha

    return nk, fermi


def get_nscf_info(nscfinfile):
    file = open(nscfinfile, "r")
    lines = file.readlines()
    for index, _ in enumerate(lines):
        line = _.split()
        if len(line) == 2 and line[0] == "K_POINTS":
            index_K_POINTS = index
            nkpt = int(lines[index_K_POINTS + 1])
        elif len(line) == 3 and line[0] == "nbnd":
            nbnd = int(line[2])
    kpts = []
    for i in range(index_K_POINTS + 2, index_K_POINTS + 2 + nkpt):
        line_split = lines[i].split()

        # Remove 1.0000000 from bands.in
        kpt_split = ""
        for j in range(3):  # x, y, z
            space = " " * (20 - len(line_split[j + 1]))
            kpt_split = kpt_split + line_split[j] + space
        kpts.append(kpt_split + "\n")

    return kpts, nbnd


def gen_pw2wan_in():
    filename = "pw2wan.in"
    file = open(filename, "w")
    file.write("&inputpp   " + "\n")
    file.write("outdir = 'out'   " + "\n")
    file.write("prefix = 'aiida'   " + "\n")
    file.write("seedname = 'aiida'   " + "\n")
    file.write("spin_component = 'none'   " + "\n")
    file.write("write_unk = .false.   " + "\n")
    file.write("reduce_unk = .true.   " + "\n")
    file.write("write_amn = .true.   " + "\n")
    file.write("write_eig = .true.   " + "\n")
    file.write("write_mmn = .true.   " + "\n")
    file.write("write_sHu = .false.    " + "\n")
    file.write("write_sIu = .false.    " + "\n")
    file.write("write_spn = .true.          " + "\n")
    file.write("spn_formatted = .true.      " + "\n")
    file.write("/   " + "\n")
    file.close()


def gen_win_in(scfxmlfile, nscfinfile, bandsinfile, bandsxmlfile):
    root = ET.parse(bandsxmlfile).getroot()
    input = root.find("input")
    gen_filename = "aiida.win"
    isExist = os.path.exists(gen_filename)
    if isExist:
        os.remove(gen_filename)
    file = open(gen_filename, "w")

    def gen_control():
        kpts, nbnd = get_nscf_info(nscfinfile)
        ex_bands, VBM, CBM = get_bands_array(
            bandsxmlfile, parent="band_structure", tag="ks_energies"
        )
        num_bands = nbnd - ex_bands
        num_wan = get_num_wan(bandsxmlfile)
        file.write("# kmesh_tol = 1e-4" + "\n")
        if ex_bands > 0:
            file.write("exclude_bands =  1 - " + str(ex_bands) + "\n")
        file.write("num_bands     =  " + str(num_bands) + "\n")
        file.write("num_wann      =  " + str(num_wan) + "\n")

        if CBM - VBM > 0:
            # This is an insulator
            dis_froz_max = CBM + 2  # eV
            file.write("# VBM = " + str(VBM) + "\n")
            file.write("# CBM = " + str(CBM) + "\n")
            file.write("# Bandgap = " + str(CBM - VBM) + "\n")
            file.write("dis_froz_max = " + str(dis_froz_max) + "\n")
            file.write("fermi_energy = " + str(VBM) + "\n")

        else:
            # This is a metal
            nk, fermi = get_scf_fermi(scfxmlfile)
            dis_froz_max = fermi + 2  # eV
            file.write("# Bandgap = " + str(CBM - VBM) + "\n")
            file.write("dis_froz_max = " + str(dis_froz_max) + "\n")
            file.write("fermi_energy = " + str(fermi) + "\n")

        file.write("# dis_win_min  = v_dis_win_min " + "\n")
        file.write("# dis_win_max  = v_dis_win_max " + "\n")
        file.write("" + "\n")

        file.write("write_tb = .true." + "\n")
        file.write("write_hr = .true." + "\n")
        file.write("write_rmn = .true." + "\n")
        file.write("write_xyz = .true." + "\n")
        file.write("spinors = .true." + "\n")
        file.write("use_ws_distance = .true." + "\n")
        file.write("spn_formatted = .true." + "\n")
        file.write("" + "\n")
        file.write("dis_num_iter      = 10000" + "\n")
        file.write("dis_conv_tol      = 1.0e-8" + "\n")
        file.write("conv_tol          = 1.0e-8" + "\n")
        file.write("conv_window       = 5" + "\n")
        file.write("num_iter          = 10000" + "\n")
        file.write("trial_step = 1.0" + "\n")
        file.write("" + "\n")
        file.write("bands_plot = true" + "\n")
        file.write("# bands_num_points = 100" + "\n")
        file.write("# restart = plot" + "\n")
        file.write("\n")

    def gen_cell():
        file.write("Begin Unit_Cell_Cart" + "\n")
        file.write("bohr" + "\n")
        for i in range(3):
            cell = input.find("atomic_structure").find("cell")
            file.write(cell[i].text + "\n")
        file.write("End Unit_Cell_Cart" + "\n")
        file.write("\n")

    def gen_atoms():
        file.write("Begin Atoms_Cart" + "\n")
        file.write("bohr" + "\n")
        nat = int(input.find("atomic_structure").get("nat"))
        atomic_positions = input.find("atomic_structure").find("atomic_positions")
        for i in range(nat):
            atomic_name = atomic_positions[i].attrib["name"]
            atomic_position = atomic_positions[i].text
            space = " " * (10 - len(atomic_name))
            file.write(atomic_name + space + atomic_position + "\n")
        file.write("End Atoms_Cart" + "\n")
        file.write("\n")

    def gen_projection():
        projections = get_projection(bandsxmlfile)
        file.write("guiding_centres = .true." + "\n")
        file.write("Begin projections" + "\n")
        for item in projections:
            file.write(str(item))
        file.write("End projections" + "\n")
        file.write("\n")

    def gen_kpath():
        kpath = gen_kpath_from_DFT(bandsinfile)
        file.write("Begin explicit_kpath" + "\n")
        for item in kpath:
            file.write(str(item))
        file.write("End explicit_kpath" + "\n")
        file.write("\n")

        file.write("Begin explicit_kpath_labels" + "\n")
        file.write(
            "GAMMA       0.0000000000       0.0000000000       0.0000000000" + "\n"
        )
        file.write("End explicit_kpath_labels" + "\n")
        file.write("\n")

    def gen_kmesh():
        kpts, nbnd = get_nscf_info(nscfinfile)
        nk, fermi = get_scf_fermi(scfxmlfile)
        nkx = nk[0]
        nky = nk[1]
        nkz = nk[2]
        if nkz != 1:
            print("WARNING: kz not equal 1, mp_grid is not consistent with nscf !")
        file.write("mp_grid = " + str(nkx) + " " + str(nky) + " 1" + "\n")
        file.write("Begin kpoints" + "\n")
        for item in kpts:
            file.write(str(item))
        file.write("End kpoints" + "\n")
        file.write("\n")

    gen_control()
    gen_cell()
    gen_atoms()
    gen_projection()
    gen_kpath()
    gen_kmesh()
    file.close()


def gen_sh(jobname):
    file = open(jobname, "w")
    file.write("#!/bin/bash" + "\n")
    file.write("#SBATCH --no-requeue" + "\n")
    file.write("#SBATCH --job-name=w90" + "\n")
    file.write("#SBATCH --partition cn" + "\n")
    file.write("#SBATCH --ntasks=24" + "\n")
    file.write("#SBATCH --nodes=1" + "\n")
    file.write("#SBATCH --ntasks-per-core=1" + "\n")
    file.write("#SBATCH --mem-per-cpu=1200 # megabytes" + "\n")
    file.write("#SBATCH --time=24:00:00" + "\n")
    file.write("#SBATCH -o slurm.%j.out        # STDOUT" + "\n")
    file.write("#SBATCH -e slurm.%j.err        # STDERR" + "\n")
    file.write("\n")
    file.write("QEpath=/home/jzhou/program/qe-7.2/" + "\n")
    file.write("W90path=/home/jzhou/program/wannier90_p2w_ham/" + "\n")
    file.write("\n")
    file.write("ln -s ../2-nscf/out" + "\n")
    file.write("source $W90path/module_load.sh" + "\n")
    file.write("$W90path/wannier90.x -pp aiida.win" + "\n")
    file.write("source $QEpath/module_load.sh" + "\n")
    file.write(
        "mpirun -n $SLURM_NTASKS $QEpath/bin/pw2wannier90.x -i pw2wan.in > pw2wan.out"
        + "\n"
    )
    file.write("source $W90path/module_load.sh" + "\n")
    file.write("mpirun -n $SLURM_NTASKS $W90path/wannier90.x aiida.win" + "\n")
    file.close()


def submit_job(scf_folder):
    abspath = scf_folder
    scfxmlfile = abspath + "/1-scf/out/aiida.xml"
    nscfinfile = abspath + "/2-nscf/aiida.in"
    bandsinfile = abspath + "/3-bands/aiida.in"
    bandsxmlfile = abspath + "/3-bands/out/aiida.xml"
    pw2wandir = abspath + "/7-pw2wan"
    isExist = os.path.exists(pw2wandir)
    if isExist:
        rmtree(pw2wandir)
    os.mkdir(pw2wandir)
    os.chdir(pw2wandir)
    gen_win_in(scfxmlfile, nscfinfile, bandsinfile, bandsxmlfile)

    gen_pw2wan_in()

    jobname = "job.sh"
    gen_sh(jobname)
    os.system("sbatch %s" % jobname)


def main():
    scf_folder = "/path/calculations/"
    submit_job(scf_folder)


if __name__ == "__main__":
    main()
