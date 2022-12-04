import click
from matrixserializer import load_matrix

@click.command()
@click.option(
    '--bin-file',
    type=click.Path(exists=True),
    required=True,
    help="Path to matrix bin file"
    )

def main(bin_file):
    matrix = load_matrix(bin_file)
    print(matrix)

if __name__ == "__main__":
    main()