from glob import glob
import subprocess
from sys import stderr

def main():
    css_files = glob('./app/**/*.css', recursive=True)

    try:
        input_css = list(filter(lambda a: 'input' in a, css_files))[0]
    except:
        print('Couldn\'t find input.css', file=stderr)
        return

    try:
        output_css = list(filter(lambda a: 'output' in a, css_files))[0]
    except:
        print('Couldn\'t find output.css', file=stderr)
        return

    subprocess.run(
        f'npx tailwindcss -i {input_css} -o {output_css} --watch'.split(' '),
        shell=True,
    )

if __name__ == '__main__':
    main()
