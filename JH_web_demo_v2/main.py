import typer
from pathlib import Path
from preprocess import resizing_cloth, resizing_human


app = typer.Typer()


@app.command()
def resize_cloth(
    image_path: Path = typer.Argument(..., help="Cloth Image files directory"),
    save_path: Path = typer.Argument(..., help="Directory where result images saved in"),
    temp_size: int = typer.Option(512, help="Size of the temp resizing before inference"),
):
    """
    Create resized cloth images and resized masked images
    """
    resizing_cloth(image_path, save_path, temp_size)


@app.command()
def test_human(
    image_file: str = typer.Argument(..., help="Cloth Image files directory"),
    temp_size: int = typer.Option(512, help="Size of the temp resizing before inference"),
):
    resized_img = resizing_human(image_file, temp_size=temp_size)
    resized_img.save(f'./test.jpg')


if __name__ == "__main__":
    app()
