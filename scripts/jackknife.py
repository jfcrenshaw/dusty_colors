from dusty_colors import selector, stacker

variants = [stacker.Default(name=f"jackknife_{i}", exclude_jk=i) for i in range(18)]

for variant in variants:
    variant.run()
