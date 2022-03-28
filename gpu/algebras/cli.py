import click
from .CGA import CGA

@click.group(chain=True)
def commands():
    pass

@commands.command()
@click.pass_context
@click.option("--size", "size", type=int)
@click.option("--base", "base_ring")
def cga(ctx, **opts):
    opts = {k: v for (k, v) in opts.items() if v is not None}
    alg = ctx.obj["latest_struct"] = CGA(**opts)
    ctx.obj["results"][alg.type_name] = alg
    return alg.bundle()


