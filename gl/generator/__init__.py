from alglbraic.util.cli import build_cli
import alglbraic
import cga

cli = build_cli(modules=[alglbraic, cga])

if __name__ == "__main__":
    cli()