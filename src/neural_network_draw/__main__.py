import sys

from .core import DrawNN


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if args:
        try:
            layers = [int(arg) for arg in args]
        except ValueError as exc:
            raise SystemExit("All layer sizes must be integers.") from exc
    else:
        layers = [3, 5, 4, 2]

    DrawNN(layers).draw()


if __name__ == "__main__":
    main()
