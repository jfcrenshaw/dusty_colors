from dusty_colors import selector, stacker

variants = [
    stacker.Default(name=f"jackknife_{i}", exclude_jk=i, bootstrap=False)
    for i in range(9)
]

for variant in variants:
    variant.run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=False,
    )
