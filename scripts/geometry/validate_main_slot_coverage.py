"""QMC sweep: verify (Upper_Corner_Notch - Upper_Corner_Ear) never overlaps slot_main.

Samples the geometry's optimizable parameters with Sobol QMC and checks, for
every sample, that the net-removed region (notch minus ear) does not eat into
slot_main. Two equivalent checks are used: direct intersection area, and
area(net_removed) + area(slot_main) == area(their union).
"""

from scipy.stats import qmc
from shapely.affinity import scale
from shapely.geometry import box
from shapely.ops import unary_union

FIXED_OFFSET = 1  # always present, never optimized away
PATCH_BRICK_2_WIDTH = 10  # constant, not optimized

N_SAMPLES_LOG2 = 10  # 2**10 = 1024
CONST_SCALE_RANGE = (0.5, 1.5)  # 50% - 150% of nominal, per user's request

# (name, nominal value)
CONST_PARAMS = [
    ("SLOT_MAIN_LENGTH", 53),
    ("SLOT_MAIN_HEIGHT", 2),
    ("PATCH_BRICK_1_SIDE_MARGIN", 6),
    ("PATCH_BRICK_1_TOP_MARGIN", 2.6),
    ("PATCH_BRICK_2_HEIGHT_MARGIN", 15),
]
K_PARAMS = [
    "UPPER_CORNER_NOTCH_1_K1",
    "UPPER_CORNER_NOTCH_1_K2",
    "UPPER_CORNER_EAR_1_K1",
    "UPPER_CORNER_EAR_1_K2",
]

MAIN_SLOT_MIN_LENGTH = 10
# NOTE for humans and agents: the manual floor below assumes the optimizer's
# search range for SLOT_MAIN_LENGTH never goes under MAIN_SLOT_MIN_LENGTH.
# If the optimizer's bounds are ever changed, re-check that assumption here.


def build_shapes(slot_len, slot_h, side_margin, top_margin, height_margin,
                  notch_k1, notch_k2, ear_k1, ear_k2):
    slot_len = max(slot_len, MAIN_SLOT_MIN_LENGTH)

    slot_main = box(-slot_len / 2, -slot_h / 2, slot_len / 2, slot_h / 2)

    ext_side = FIXED_OFFSET + side_margin
    ext_up = FIXED_OFFSET + top_margin
    slot_min_x, slot_min_y, slot_max_x, slot_max_y = slot_main.bounds
    patch_brick_1 = box(
        slot_min_x - ext_side, slot_min_y,
        slot_max_x + ext_side, slot_max_y + ext_up,
    )

    patch_brick_2_height = top_margin + height_margin
    patch_brick_2 = box(
        -PATCH_BRICK_2_WIDTH / 2, slot_max_y,
        PATCH_BRICK_2_WIDTH / 2, slot_max_y + patch_brick_2_height,
    )

    all_shapes = (slot_main, patch_brick_1, patch_brick_2)
    sub_min_x = min(s.bounds[0] for s in all_shapes)
    sub_max_x = max(s.bounds[2] for s in all_shapes)
    sub_max_y = max(s.bounds[3] for s in all_shapes)
    substrate_width = sub_max_x - sub_min_x

    notch_width = -(substrate_width - PATCH_BRICK_2_WIDTH) / 2 * notch_k1
    notch_height = -height_margin * notch_k2
    notch_1 = box(
        sub_max_x + notch_width, sub_max_y + notch_height,
        sub_max_x, sub_max_y,
    )

    x1, y1 = notch_1.bounds[0], notch_1.bounds[1]
    length = -notch_width
    height = -notch_height
    ear_width = ear_k1 * length
    ear_height = ear_k2 * height
    ear_1 = box(x1, y1, x1 + ear_width, y1 + ear_height)

    notch_2 = scale(notch_1, xfact=-1, yfact=1, origin=(0, 0))
    ear_2 = scale(ear_1, xfact=-1, yfact=1, origin=(0, 0))

    return slot_main, notch_1, notch_2, ear_1, ear_2


def main():
    dims = len(CONST_PARAMS) + len(K_PARAMS)
    sampler = qmc.Sobol(d=dims, scramble=True, seed=42)
    unit_samples = sampler.random_base2(m=N_SAMPLES_LOG2)

    const_lo = [nominal * CONST_SCALE_RANGE[0] for _, nominal in CONST_PARAMS]
    const_hi = [nominal * CONST_SCALE_RANGE[1] for _, nominal in CONST_PARAMS]
    l_bounds = const_lo + [0.0] * len(K_PARAMS)
    u_bounds = const_hi + [1.0] * len(K_PARAMS)
    samples = qmc.scale(unit_samples, l_bounds, u_bounds)

    max_overlap_area = 0.0
    max_area_mismatch = 0.0
    worst_sample = None

    for row in samples:
        slot_main, notch_1, notch_2, ear_1, ear_2 = build_shapes(*row)

        notch_union = unary_union([notch_1, notch_2])
        ear_union = unary_union([ear_1, ear_2])
        net_removed = notch_union.difference(ear_union)

        overlap_area = net_removed.intersection(slot_main).area
        sum_areas = net_removed.area + slot_main.area
        union_area = net_removed.union(slot_main).area
        area_mismatch = abs(sum_areas - union_area)

        if overlap_area > max_overlap_area:
            max_overlap_area = overlap_area
            worst_sample = row
        max_area_mismatch = max(max_area_mismatch, area_mismatch)

    n_samples = len(samples)
    print(f"Samples checked: {n_samples}")
    print(f"Max (notch - ear) / slot_main intersection area: {max_overlap_area:.3e}")
    print(f"Max |sum(areas) - union(area)| mismatch: {max_area_mismatch:.3e}")
    if worst_sample is not None:
        names = [n for n, _ in CONST_PARAMS] + K_PARAMS
        print("Worst-case sample:")
        for name, value in zip(names, worst_sample):
            print(f"  {name} = {value}")

    ok = max_overlap_area < 1e-9 and max_area_mismatch < 1e-9
    print("PASS: main_slot is always fully covered" if ok
          else "FAIL: main_slot gets cut into for at least one sample")


if __name__ == "__main__":
    main()
