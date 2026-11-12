"""Main-group coordination labels and bond orders for the preset force field.

The model is deliberately coarse. It only needs a stable bond-order label for
the preset terms, not a valence-bond calculation.
"""

from __future__ import annotations

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
    "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
    "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

SYMBOL_TO_Z = {symbol: number for number, symbol in enumerate(_SYMBOLS, start=1)}

# Electrons available for covalent bonding, used as the "unpaired" count.
UNPAIRED_ELECTRONS = {
    "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
    "Al": 3, "Si": 4, "P": 3, "S": 2, "Cl": 1,
    "Ga": 3, "Ge": 4, "As": 3, "Se": 2, "Br": 1,
    "In": 3, "Sn": 4, "Sb": 3, "Te": 2, "I": 1,
    "Tl": 3, "Pb": 4, "Bi": 3, "Po": 2, "At": 1,
}

# Coordination numbers above the unpaired-electron count.
HYPERVALENT_GEOMETRIES = {
    5: "tbp",
    6: "oct",
    7: "pbp",
    8: "cubic",
}

_BOND_ORDERS = {
    "sp": 3,
    "sp2": 2,
    "sp3": 1,
    "tbp": 1,
    "oct": 1,
    "pbp": 1,
    "cubic": 1,
    "unknown": 1,
}

# Ideal angles in degrees for each coordination geometry.
IDEAL_ANGLES = {
    "sp": (180.0,),
    "sp2": (120.0,),
    "sp3": (109.47122063449069,),
    "tbp": (90.0, 120.0, 180.0),
    "oct": (90.0, 180.0),
    "pbp": (72.0, 90.0, 180.0),
    "cubic": (70.52877936550931, 109.47122063449069, 180.0),
    "unknown": (),
}


def geometry_label(degree: int, symbol: str) -> str:
    """Return the coordination geometry label for one main-group atom.

    Args:
        degree: Number of unique bonded neighbours.
        symbol: Element symbol.

    Return:
        ``sp``, ``sp2``, ``sp3``, ``tbp``, ``oct``, ``pbp``, ``cubic`` or
        ``unknown`` for elements outside the main-group table.
    """
    degree = int(degree)
    if degree in HYPERVALENT_GEOMETRIES:
        return HYPERVALENT_GEOMETRIES[degree]
    unpaired = UNPAIRED_ELECTRONS.get(str(symbol))
    if unpaired is None:
        return "unknown"
    difference = unpaired - degree
    if difference >= 2:
        return "sp"
    if difference == 1:
        return "sp2"
    return "sp3"


def bond_order(first_label: str, second_label: str) -> int:
    """Return the pair bond order from two geometry labels.

    Args:
        first_label: Geometry label of the first atom.
        second_label: Geometry label of the second atom.

    Return:
        The smaller of the two label orders, which also covers equal labels.
    """
    return min(
        _BOND_ORDERS.get(str(first_label), 1),
        _BOND_ORDERS.get(str(second_label), 1),
    )


def ideal_angles(label: str) -> tuple[float, ...]:
    """Return the ideal angles in degrees for one geometry label."""
    return IDEAL_ANGLES.get(str(label), ())
