"""Shopee-style parcel labels with an ArUco marker (instead of a QR code).

One A4 page per label size (default 6, 5 and 4 cm squares). Each label
has fake order, sender and recipient details, a Code 128 tracking
barcode and a unique ArUco marker (DICT_4X4_100, one ID per label).
labels.csv maps each marker ID to its size and tracking number.
parcel_labels_test_A4.pdf has only the first label of each size, for
a real camera test (test_aruco_live.py).

Print at 100% scale ("actual size", no "fit to page"). Check the 50 mm
ruler on each page with a real ruler after printing.

Usage:
  python outputs/portfolio/parcel_labels/make_labels.py
  python outputs/portfolio/parcel_labels/make_labels.py --sizes 4,5,6
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
from reportlab.graphics.barcode import code128
from reportlab.lib.colors import black, Color
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
DICT = cv2.aruco.DICT_4X4_100
GREY = Color(0.6, 0.6, 0.6)
MARGIN, GAP = 10 * mm, 4 * mm

# fictional names, streets, towns and postcodes only (no real people or
# addresses); postcodes 00xxx / 99xxx are not used in Malaysia
NAMES = ["SpongeBob SquarePants", "Patrick Star", "Squidward Tentacles",
         "Shrek Swampington", "Doge McWowface", "Big Chungus",
         "Nyan Kat", "Captain Stonks", "Sir Bonks-a-Lot", "Meme Man",
         "Pepe Frogsworth", "Chad Gigachad", "Kucing Oren",
         "Encik Ayam Goreng", "Teh Tarik Bro", "Durian Supreme",
         "Roti Canai Jr", "Professor Bruh", "Lord Yeetington",
         "Sus Amogus", "Grumpy Toaster", "Duck of Destiny"]
STREETS = ["Conch Street", "Jalan 404 Not Found", "Lorong Stonks Naik",
           "Jalan Bonk Bonk", "Persiaran Kucing Oren", "Swamp Lane",
           "Lebuh Wifi Percuma", "Jalan Mee Segera", "Lorong Amogus",
           "Jalan Much Wow", "Persiaran Rickroll"]
AREAS = [("00404", "Bikini Bottom"), ("99999", "Far Far Away"),
         ("00000", "Kota Stonks"), ("00001", "Pulau Doge"),
         ("99998", "Bandar Meme"), ("00420", "Taman Chungus")]
PHONE_ENDS = ["404", "123", "777", "000", "999", "321"]
COURIER = "SO-101 EXPRESS"


def fake_details(rng, marker_id):
    def phone():  # 6000... is not a valid Malaysian number
        return f"6000*****{rng.choice(PHONE_ENDS)}"

    def address():
        code, city = rng.choice(AREAS)
        return (f"No. {rng.randint(1, 199)}, {rng.choice(STREETS)}",
                f"{code} {city}")

    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
                      for _ in range(4))
    sender, recipient = rng.sample(NAMES, 2)
    return {
        "tracking": f"SPXMY{rng.randint(10**9, 10**10 - 1)}",
        "order": f"2610{rng.randint(10, 31)}"
                 f"{rng.randint(10**5, 10**6 - 1)}{letters}",
        "sender": sender, "sender_addr": address(),
        "sender_phone": phone(),
        "recipient": recipient, "recipient_addr": address(),
        "recipient_phone": phone(),
        "weight": f"{rng.uniform(0.1, 2.0):.2f} kg",
        "qty": rng.randint(1, 3),
        "ship_by": f"{rng.randint(28, 31)}-10-2026",
        "marker_id": marker_id,
    }


def draw_aruco(c, x, y, size, marker_id):
    """Vector ArUco marker (with its black border) at (x, y), size pts."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
    cell = size / 6
    for r in range(6):
        for col in range(6):
            if bits[r, col] == 0:
                c.rect(x + col * cell, y + (5 - r) * cell, cell, cell,
                       stroke=0, fill=1)


def draw_label(c, x, y, size, d):
    """One square label; layout designed at 60 mm and scaled by k."""
    k = size / (60 * mm)

    def u(v):  # design mm -> points
        return v * mm * k

    def text(tx, ty, s, pt, bold=False, color=black, max_w=None):
        """max_w (design mm): shrink the font so the text fits."""
        font = "Helvetica-Bold" if bold else "Helvetica"
        size_pt = pt * k
        if max_w is not None:
            width = stringWidth(s, font, size_pt)
            size_pt *= min(1.0, u(max_w) / width)
        c.setFillColor(color)
        c.setFont(font, size_pt)
        c.drawString(x + u(tx), y + u(ty), s)

    c.saveState()
    c.setLineWidth(0.25)
    c.setStrokeColor(GREY)  # cut line
    c.rect(x, y, size, size, stroke=1, fill=0)
    c.setStrokeColor(black)
    c.setLineWidth(max(0.5 * k, 0.3))
    c.rect(x + u(1), y + u(1), u(58), u(58), stroke=1, fill=0)

    # header: no solid black box, it reads as a false ArUco marker
    text(3, 53.6, COURIER, 9, bold=True)
    text(44, 53.8, "STANDARD", 5.5, bold=True)
    c.setLineWidth(max(1.2 * k, 0.6))
    c.line(x + u(1), y + u(51.5), x + u(59), y + u(51.5))
    c.setLineWidth(max(0.5 * k, 0.3))

    # tracking barcode, fitted to 54 mm
    bc = code128.Code128(d["tracking"], barHeight=u(8), barWidth=1,
                         quiet=False)
    bc = code128.Code128(d["tracking"], barHeight=u(8), quiet=False,
                         barWidth=u(54) / bc.width)
    c.setFillColor(black)
    bc.drawOn(c, x + u(3), y + u(42))
    c.setFont("Helvetica-Bold", 5.5 * k)
    c.drawCentredString(x + u(30), y + u(39.6), d["tracking"])
    c.line(x + u(1), y + u(38.5), x + u(59), y + u(38.5))

    # sender / recipient (left column)
    text(2.5, 35.6, "FROM", 4.5, bold=True)
    text(10, 35.6, d["sender"], 5, max_w=24)
    text(2.5, 33.3, d["sender_addr"][0], 4, max_w=31)
    text(2.5, 31.3, f"{d['sender_addr'][1]}  {d['sender_phone']}", 4,
         max_w=31)
    c.line(x + u(1), y + u(29.8), x + u(34), y + u(29.8))
    text(2.5, 26.6, "TO", 4.5, bold=True)
    text(10, 26.6, d["recipient"], 6, bold=True, max_w=24)
    text(2.5, 23.8, d["recipient_addr"][0], 4.5, max_w=31)
    text(2.5, 21.4, d["recipient_addr"][1], 4.5, max_w=31)
    text(2.5, 19.0, f"Tel: {d['recipient_phone']}", 4.5, max_w=31)

    # ArUco marker (right column) with a white quiet zone around it
    c.setFillColor(black)
    draw_aruco(c, x + u(37), y + u(15.5), u(19.5), d["marker_id"])
    text(37.5, 12.3, f"ArUco 4x4  ID {d['marker_id']:02d}", 4)

    # order details
    c.line(x + u(1), y + u(16.8), x + u(34), y + u(16.8))
    c.line(x + u(1), y + u(11.2), x + u(59), y + u(11.2))
    text(2.5, 14.2, "ORDER DETAILS", 4.5, bold=True)
    text(2.5, 8.3, f"Order ID: {d['order']}", 4.5, bold=True)
    text(2.5, 5.6, f"Weight: {d['weight']}   Qty: {d['qty']}   COD: No",
         4.5)
    text(2.5, 2.9, f"Ship by: {d['ship_by']}   Parcel {size / (10 * mm):.0f}"
         " cm label", 4.5)
    c.restoreState()


def draw_ruler(c, page_w):
    """50 mm ruler to check the print scale."""
    x0, y0 = MARGIN, 5 * mm
    c.setLineWidth(0.4)
    c.line(x0, y0, x0 + 50 * mm, y0)
    for i in range(51):
        h = 2.5 * mm if i % 10 == 0 else (1.6 * mm if i % 5 == 0 else mm)
        c.line(x0 + i * mm, y0, x0 + i * mm, y0 + h)
    c.setFont("Helvetica", 6)
    c.drawString(x0 + 52 * mm, y0, "50 mm: print at 100% (actual size)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="6,5,4", help="label sizes in cm")
    ap.add_argument("--out", default=str(HERE / "parcel_labels_A4.pdf"))
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    page_w, page_h = A4
    c = canvas.Canvas(args.out, pagesize=A4)
    c.setTitle("Parcel labels with ArUco markers")
    rows_csv, marker_id, firsts = [], 0, []
    for size_cm in (float(s) for s in args.sizes.split(",")):
        size = size_cm * 10 * mm
        cols = int((page_w - 2 * MARGIN + GAP) // (size + GAP))
        rows = int((page_h - 2 * MARGIN - 8 * mm + GAP) // (size + GAP))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, page_h - MARGIN + 2 * mm,
                     f"Parcel labels {size_cm:g} x {size_cm:g} cm "
                     f"(ArUco DICT_4X4_100)")
        for r in range(rows):
            for col in range(cols):
                if marker_id >= 100:
                    raise SystemExit("DICT_4X4_100 has only 100 IDs")
                d = fake_details(rng, marker_id)
                x = MARGIN + col * (size + GAP)
                y = page_h - MARGIN - 6 * mm - (r + 1) * size - r * GAP
                draw_label(c, x, y, size, d)
                if r == 0 and col == 0:
                    firsts.append((size, d))
                rows_csv.append({"marker_id": marker_id,
                                 "size_cm": size_cm,
                                 "tracking": d["tracking"],
                                 "order": d["order"],
                                 "page": c.getPageNumber()})
                marker_id += 1
        draw_ruler(c, page_w)
        c.showPage()
    c.save()
    with open(Path(args.out).with_name("labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"saved {args.out} ({len(rows_csv)} labels)")
    test = Path(args.out).with_name("parcel_labels_test_A4.pdf")
    make_test_page(test, firsts)
    print(f"saved {test} (one label per size: IDs "
          f"{', '.join(str(d['marker_id']) for _, d in firsts)})")


def make_test_page(path, labels):
    """One A4 page with one label of each size, in a row."""
    page_w, page_h = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Parcel label test page")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, page_h - MARGIN + 2 * mm,
                 "Parcel label test: one label per size "
                 "(ArUco DICT_4X4_100)")
    x = MARGIN
    for size, d in labels:
        draw_label(c, x, page_h - MARGIN - 6 * mm - size, size, d)
        x += size + 2 * GAP
    draw_ruler(c, page_w)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
