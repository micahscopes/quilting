from alglbraic.util.cli import build_cli
import alglbraic
from .algebras import cli as cga

cli = build_cli(modules=[alglbraic, cga])

if __name__ == "__main__":
    cli()