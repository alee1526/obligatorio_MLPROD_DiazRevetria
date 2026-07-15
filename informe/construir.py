"""Construye el informe como un unico archivo HTML autocontenido.

Toma fuente.html (que referencia estilo.css y img/) y produce informe.html con
el CSS embebido en un <style> y cada imagen convertida a data URI. El resultado
no depende de ningun archivo vecino: se puede mover, mandar por mail o meter en
el zip de la entrega y sigue viendose igual.

Uso:  python informe/construir.py
"""
import base64
import re
from pathlib import Path

AQUI = Path(__file__).parent
FUENTE = AQUI / "fuente.html"
SALIDA = AQUI / "informe.html"
CSS = AQUI / "estilo.css"

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def data_uri(ruta: Path) -> str:
    b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:{MIME[ruta.suffix.lower()]};base64,{b64}"


def main():
    html = FUENTE.read_text(encoding="utf-8")

    # 1. CSS embebido
    html = html.replace(
        '<link rel="stylesheet" href="estilo.css">',
        "<style>\n" + CSS.read_text(encoding="utf-8") + "\n</style>",
    )

    # 2. Los SVG van inline: quedan como vectores y escalan sin pixelarse.
    def inline_svg(m):
        ruta = AQUI / m.group(1)
        svg = ruta.read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml[^>]*\?>", "", svg).strip()
        return svg

    html = re.sub(
        r'<object data="(img/[^"]+\.svg)"[^>]*></object>', inline_svg, html
    )

    # 3. El resto de las imagenes, a data URI
    incrustadas = []

    def incrustar(m):
        antes, ruta_rel, despues = m.group(1), m.group(2), m.group(3)
        ruta = AQUI / ruta_rel
        incrustadas.append(ruta.name)
        return f'<img {antes}src="{data_uri(ruta)}"{despues}>'

    html = re.sub(r'<img ([^>]*?)src="(img/[^"]+)"([^>]*?)>', incrustar, html)

    SALIDA.write_text(html, encoding="utf-8")

    faltan = re.findall(r'(?:src|data)="(img/[^"]+)"', html)
    print(f"imagenes embebidas: {len(incrustadas)}")
    print(f"referencias externas que quedaron: {len(faltan)} {faltan if faltan else ''}")
    print(f"-> {SALIDA}  ({SALIDA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
