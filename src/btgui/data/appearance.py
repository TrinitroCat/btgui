"""Default element colors and radii for the renderer.

The radius JSON files are kept as editable data stubs. Covalent values are in
picometres and are converted to angstroms when returned to the renderer.
"""

import json
from pathlib import Path


_DATA_DIR = Path(__file__).parent
_COVALENT = json.loads((_DATA_DIR / "covar_radii_stub.json").read_text(encoding="utf-8"))
_VDW = json.loads((_DATA_DIR / "vdw_radii_stub.json").read_text(encoding="utf-8"))

ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg"
).split()

ELEMENT_COLORS = {
    "H": "#f2f2f2", "He": "#d9ffff", "Li": "#cc80ff", "Be": "#c2ff00", "B": "#ffb5b5",
    "C": "#4a4a4a", "N": "#3050f8", "O": "#ff2020", "F": "#90e050", "Ne": "#b3e3f5",
    "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6", "Si": "#f0c8a0", "P": "#ff8000",
    "S": "#ffff30", "Cl": "#1ff01f", "Ar": "#80d1e3", "K": "#8f40d4", "Ca": "#3dff00",
    "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0",
    "Br": "#a62929", "I": "#940094", "Au": "#ffd123", "Hg": "#b8b8d0",
}
_SYMBOL_TO_NAME = dict(item.split(":") for item in (
    "H:Hydrogen He:Helium "
    "Li:Lithium Be:Beryllium B:Boron C:Carbon N:Nitrogen O:Oxygen F:Fluorine Ne:Neon "
    "Na:Sodium Mg:Magnesium Al:Aluminum Si:Silicon P:Phosphorus S:Sulfur Cl:Chlorine Ar:Argon "
    "K:Potassium Ca:Calcium "
    "Sc:Scandium Ti:Titanium V:Vanadium Cr:Chromium Mn:Manganese Fe:Iron Co:Cobalt Ni:Nickel Cu:Copper Zn:Zinc "
    "Ga:Gallium Ge:Germanium As:Arsenic Se:Selenium Br:Bromine Kr:Krypton "
    "Rb:Rubidium Sr:Strontium "
    "Y:Yttrium Zr:Zirconium Nb:Niobium Mo:Molybdenum Tc:Technetium Ru:Ruthenium Rh:Rhodium Pd:Palladium Ag:Silver Cd:Cadmium "
    "In:Indium Sn:Tin Sb:Antimony Te:Tellurium I:Iodine Xe:Xenon "
    "Cs:Cesium Ba:Barium "
    "La:Lanthanum Ce:Cerium Pr:Praseodymium Nd:Neodymium Pm:Promethium Sm:Samarium Eu:Europium Gd:Gadolinium Tb:Terbium Dy:Dysprosium Ho:Holmium Er:Erbium Tm:Thulium Yb:Ytterbium Lu:Lutetium "
    "Hf:Hafnium Ta:Tantalum W:Tungsten Re:Rhenium Os:Osmium Ir:Iridium Pt:Platinum Au:Gold "
    "Hg:Mercury Tl:Thallium Pb:Lead Bi:Bismuth Po:Polonium At:Astatine Rn:Radon "
    "Fr:Francium Ra:Radium "
    "Ac:Actinium Th:Thorium Pa:Protactinium U:Uranium Np:Neptunium Pu:Plutonium Am:Americium Cm:Curium Bk:Berkelium Cf:Californium Es:Einsteinium Fm:Fermium Md:Mendelevium No:Nobelium Lr:Lawrencium "
    "Rf:Rutherfordium Db:Dubnium Sg:Seaborgium Bh:Bohrium Hs:Hassium Mt:Meitnerium Ds:Darmstadtium Rg:Roentgenium".split())
                       )


def _numeric_radius(value, fallback):
    """Choose a numeric radius from a scalar or a state-specific mapping.

    Args:
        value: JSON radius value.
        fallback: Radius used when no numeric value is available.

    Return:
        Radius in angstroms.
    """
    if isinstance(value, dict):
        for key in ("sp3", "low_spin", "high_spin"):
            if isinstance(value.get(key), (float, int)):
                return float(value[key]) / 100.0
        values = [item for item in value.values() if isinstance(item, (float, int))]
        return (float(values[0]) / 100.0) if values else fallback
    return (float(value) / 100.0) if isinstance(value, (float, int)) else fallback


def covalent_radius(symbol):
    """Return a covalent radius in angstroms for an element symbol."""
    key = _SYMBOL_TO_NAME.get(symbol, symbol)
    return _numeric_radius(_COVALENT.get(key), 0.76)


def vdw_radius(symbol):
    """Return a van der Waals radius in angstroms for an element symbol."""
    return _numeric_radius(_VDW.get(symbol), 1.5)


def default_color(symbol):
    """Return a hex color for an element symbol."""
    return ELEMENT_COLORS.get(symbol, "#5a9b72")
