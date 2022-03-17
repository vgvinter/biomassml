import click


@click.command("cli")
@click.argument("kernel_name", type=click.Choice(["coreg", "rbf"]))
def test(kernel_name):
    print(kernel_name)


if __name__ == "__main__":
    test()
