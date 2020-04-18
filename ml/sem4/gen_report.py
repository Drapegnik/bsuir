#!/usr/bin/env python3

num_labs = 8

with open('./report.md', 'w') as report:
    with open('./head.md', 'r') as head:
        report.write(head.read())

    for i in range(1, num_labs + 1):
        dir = f'lab{i}'

        with open(f'{dir}/readme.md', 'r') as readme:
            report.write(readme.read())

        with open(f'{dir}/{dir}.md', 'r') as lab:
            lab_report = lab.read()
            # fix images path
            report.write(
                lab_report.replace('](./out/output', f'](./{dir}/out/output').replace('src="./out', f'src="./{dir}/out')
            )

        print(f'write {dir}')
