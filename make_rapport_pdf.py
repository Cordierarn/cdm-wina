"""
Génère rapport.pdf — CdM 2026, modèle statistique
2 pages : page 1 = favoris + méthodo, page 2 = groupes
"""

import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

W, H = A4  # 595 x 842 pt

# ── Palette ───────────────────────────────────────────────
NAVY   = HexColor('#0B1120')
NAVY2  = HexColor('#141C2B')
GOLD   = HexColor('#F2B705')
GOLD2  = HexColor('#FFD045')
GREEN  = HexColor('#22C55E')
RED    = HexColor('#EF4444')
GRAY   = HexColor('#64748B')
LGRAY  = HexColor('#CBD5E1')
OFFWH  = HexColor('#F8FAFF')
WHITE  = HexColor('#FFFFFF')
SLATE  = HexColor('#1E293B')
MUTED  = HexColor('#94A3B8')

# ── Data ──────────────────────────────────────────────────
FAVORITES = [
    ("Argentine",   "🇦🇷", 0.954, 15.5, 24.5, 37.1, 52.8, 68.4, 98.5),
    ("Espagne",     "🇪🇸", 0.889, 14.9, 23.8, 35.3, 48.1, 70.7, 99.2),
    ("France",      "🇫🇷", 0.846,  9.6, 16.8, 29.2, 45.0, 69.0, 95.6),
    ("Angleterre",  "🏴",  0.821,  7.8, 14.0, 25.0, 41.6, 65.3, 98.2),
    ("Brésil",      "🇧🇷", 0.806,  5.7, 11.0, 20.8, 36.1, 59.5, 96.4),
    ("Colombie",    "🇨🇴", 0.775,  5.3, 10.2, 18.8, 33.9, 59.3, 94.4),
    ("Portugal",    "🇵🇹", 0.772,  5.3, 10.3, 19.5, 36.3, 61.1, 94.6),
    ("Allemagne",   "🇩🇪", 0.764,  4.1,  8.5, 17.3, 31.1, 59.4, 95.9),
]
# name, flag, strength, p_win, p_final, p_semi, p_qf, p_r16, p_q

GROUPS_SHORT = [
    ("A", [("Mexique","🇲🇽",88.8), ("Corée du Sud","🇰🇷",79.6),
           ("Rép. Tchèque","🇨🇿",75.6), ("Afrique du Sud","🇿🇦",31.2)]),
    ("B", [("Suisse","🇨🇭",94.5), ("Canada","🇨🇦",89.3),
           ("Bosnie","🇧🇦",53.1), ("Qatar","🇶🇦",28.3)]),
    ("C", [("Brésil","🇧🇷",96.4), ("Maroc","🇲🇦",87.2),
           ("Écosse","🏴",77.2), ("Haïti","🇭🇹",12.9)]),
    ("D", [("Turquie","🇹🇷",90.0), ("Australie","🇦🇺",67.4),
           ("États-Unis","🇺🇸",64.3), ("Paraguay","🇵🇾",52.0)]),
    ("E", [("Allemagne","🇩🇪",95.9), ("Équateur","🇪🇨",92.3),
           ("Côte d'Ivoire","🇨🇮",81.5), ("Curaçao","🇨🇼",8.2)]),
    ("F", [("Pays-Bas","🇳🇱",92.5), ("Japon","🇯🇵",88.1),
           ("Suède","🇸🇪",56.8), ("Tunisie","🇹🇳",31.2)]),
    ("G", [("Belgique","🇧🇪",93.7), ("Iran","🇮🇷",83.1),
           ("Égypte","🇪🇬",68.4), ("Nv.-Zélande","🇳🇿",25.7)]),
    ("H", [("Espagne","🇪🇸",99.2), ("Uruguay","🇺🇾",91.2),
           ("Cap-Vert","🇨🇻",33.0), ("Arabie S.","🇸🇦",21.2)]),
    ("I", [("France","🇫🇷",95.6), ("Sénégal","🇸🇳",84.5),
           ("Norvège","🇳🇴",83.7), ("Irak","🇮🇶",11.8)]),
    ("J", [("Argentine","🇦🇷",98.5), ("Autriche","🇦🇹",75.6),
           ("Algérie","🇩🇿",61.3), ("Jordanie","🇯🇴",25.2)]),
    ("K", [("Portugal","🇵🇹",94.6), ("Colombie","🇨🇴",94.4),
           ("Ouzbékistan","🇺🇿",47.6), ("RD Congo","🇨🇩",22.9)]),
    ("L", [("Angleterre","🏴",98.2), ("Croatie","🇭🇷",94.1),
           ("Panama","🇵🇦",47.4), ("Ghana","🇬🇭",14.9)]),
]

LIVE = [
    ("Mexique", "Afrique du Sud", "2 – 0"),
    ("Corée du Sud", "Rép. Tchèque", "2 – 1"),
    ("Canada", "Bosnie-Herz.", "1 – 1"),
    ("États-Unis", "Paraguay", "4 – 1"),
    ("Qatar", "Suisse", "1 – 1"),
    ("Brésil", "Maroc", "1 – 1"),
    ("Haïti", "Écosse", "0 – 1"),
    ("Australie", "Turquie", "2 – 0"),
]


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def set_fill(c, col):   c.setFillColor(col)
def set_stroke(c, col): c.setStrokeColor(col)

def rect(c, x, y, w, h, fill=None, stroke=None, radius=0):
    if fill:   c.setFillColor(fill)
    if stroke: c.setStrokeColor(stroke)
    if radius:
        c.roundRect(x, y, w, h, radius, fill=1 if fill else 0, stroke=1 if stroke else 0)
    else:
        c.rect(x, y, w, h, fill=1 if fill else 0, stroke=1 if stroke else 0)

def text(c, x, y, s, size=10, bold=False, col=None, align='left'):
    if col:  c.setFillColor(col)
    fn = 'Helvetica-Bold' if bold else 'Helvetica'
    c.setFont(fn, size)
    if align == 'center':
        c.drawCentredString(x, y, s)
    elif align == 'right':
        c.drawRightString(x, y, s)
    else:
        c.drawString(x, y, s)

def label(c, x, y, s, size=7, col=None, spacing=2):
    if col: c.setFillColor(col)
    c.setFont('Helvetica-Bold', size)
    # letter spacing via individual chars
    cx = x
    for ch in s.upper():
        c.drawString(cx, y, ch)
        cx += c.stringWidth(ch, 'Helvetica-Bold', size) + spacing

def bar_h(c, x, y, w, h, pct, fill_col, bg_col=None, radius=2):
    if bg_col:
        rect(c, x, y, w, h, fill=bg_col, radius=radius)
    filled_w = w * pct / 100
    if filled_w > 0:
        rect(c, x, y, max(filled_w, radius*2), h, fill=fill_col, radius=radius)

def matplotlib_to_reader(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor='none', transparent=True)
    buf.seek(0)
    return ImageReader(buf)


# ═══════════════════════════════════════════════════════════
#  PAGE 1
# ═══════════════════════════════════════════════════════════

def draw_page1(c):
    # ── Background ─────────────────────────────────────────
    rect(c, 0, 0, W, H, fill=OFFWH)

    # ── HEADER BAND ────────────────────────────────────────
    header_h = 108
    rect(c, 0, H - header_h, W, header_h, fill=NAVY)

    # Decorative circle (centre du terrain)
    c.setStrokeColor(HexColor('#ffffff08'))
    c.setLineWidth(0.5)
    c.circle(W/2, H - header_h/2, 65, fill=0, stroke=1)
    c.circle(W/2, H - header_h/2, 3, fill=0, stroke=1)

    # Title
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 30)
    c.drawString(22*mm, H - 42, 'COUPE DU MONDE')
    c.setFillColor(GOLD)
    c.drawString(22*mm, H - 68, '2026')
    c.setFillColor(HexColor('#94A3B8'))
    c.setFont('Helvetica', 9)
    c.drawString(22*mm, H - 84, 'MODÈLE STATISTIQUE PERSONNEL  ·  50 000 SIMULATIONS  ·  48 ÉQUIPES')

    # Stats top right
    stats = [('50k', 'Simulations'), ('378k', 'Matchs sources'), ('1,27', 'Buts/équipe base')]
    sx = W - 22*mm
    for val, lbl in reversed(stats):
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 18)
        c.drawRightString(sx, H - 44, val)
        c.setFillColor(HexColor('#64748B'))
        c.setFont('Helvetica', 7)
        c.drawRightString(sx, H - 57, lbl.upper())
        sx -= 52

    # Live dot
    c.setFillColor(GREEN)
    c.circle(W - 22*mm, H - 78, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 8)
    c.drawString(W - 20*mm - 2, H - 81, 'TOURNOI EN COURS · DÉMARRÉ LE 11 JUIN')

    y_cur = H - header_h - 10

    # ── LIVE RESULTS RIBBON ────────────────────────────────
    ribbon_h = 22
    rect(c, 0, y_cur - ribbon_h, W, ribbon_h, fill=NAVY2)
    rect(c, 0, y_cur - ribbon_h, 38, ribbon_h, fill=GREEN)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 7)
    c.drawString(4, y_cur - ribbon_h + 7, 'LIVE')
    c.drawString(4, y_cur - ribbon_h + 16, 'J1')

    # Results
    rx = 46
    for home, away, score in LIVE:
        c.setFillColor(HexColor('#CBD5E1'))
        c.setFont('Helvetica', 7)
        c.drawString(rx, y_cur - 8, home[:8])
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 9)
        c.drawCentredString(rx + 40, y_cur - 7, score)
        c.setFillColor(HexColor('#CBD5E1'))
        c.setFont('Helvetica', 7)
        c.drawString(rx + 52, y_cur - 8, away[:8])
        rx += 88
        if rx > W - 30:
            break

    y_cur -= ribbon_h + 14

    # ── SECTION TITLE : LES FAVORIS ───────────────────────
    c.setFillColor(GOLD)
    c.setFont('Helvetica-Bold', 7)
    c.drawString(22*mm, y_cur, 'PROBABILITÉS DE REMPORTER LE TITRE')
    y_cur -= 14
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 18)
    c.drawString(22*mm, y_cur, 'Les favoris')
    y_cur -= 6

    # Divider
    c.setStrokeColor(HexColor('#E2E8F0'))
    c.setLineWidth(0.5)
    c.line(22*mm, y_cur, W - 22*mm, y_cur)
    y_cur -= 10

    # Columns headers
    cols = {'rank': 22*mm, 'name': 32*mm, 'funnel': 90*mm, 'pct': W - 22*mm}
    stage_labels = ['GRP', '8E', 'QF', 'SF', 'FIN', '🏆']
    stage_x = [90*mm + i*13 for i in range(6)]

    c.setFillColor(MUTED)
    c.setFont('Helvetica-Bold', 6)
    for i, lbl in enumerate(stage_labels):
        c.drawCentredString(stage_x[i] + 5, y_cur, lbl)
    c.drawRightString(cols['pct'], y_cur, 'TITRE')
    y_cur -= 6

    # ── Team rows ──────────────────────────────────────────
    row_h = 28
    rank_colors = [GOLD, HexColor('#94A3B8'), HexColor('#CD7F32')]

    for i, (name, flag, strength, p_win, p_final, p_semi, p_qf, p_r16, p_q) in enumerate(FAVORITES):
        bg = HexColor('#F8FAFF') if i % 2 == 0 else WHITE
        rect(c, 18*mm, y_cur - row_h + 4, W - 36*mm, row_h, fill=bg, radius=3)

        # Rank
        rc = rank_colors[i] if i < 3 else LGRAY
        c.setFillColor(rc)
        c.setFont('Helvetica-Bold', 14)
        c.drawCentredString(25*mm, y_cur - 10, str(i+1))

        # Name
        c.setFillColor(SLATE)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(32*mm, y_cur - 8, name)

        # Strength bar (thin, under name)
        bar_h(c, 32*mm, y_cur - 16, 45, 3, strength*100, GOLD, HexColor('#E2E8F0'), 2)

        # Stage dots (filled circles proportional to probability)
        stage_probs = [p_q, p_r16, p_qf, p_semi, p_final, p_win]
        stage_cols  = [GREEN, GOLD, GOLD, GOLD, GOLD, GOLD]
        for j, (prob, scol) in enumerate(zip(stage_probs, stage_cols)):
            sx = stage_x[j] + 5
            sy = y_cur - 10
            # background circle
            c.setFillColor(HexColor('#E2E8F0'))
            c.circle(sx, sy, 6, fill=1, stroke=0)
            # filled arc proportional to prob
            alpha = int(55 + 200 * (prob/100))
            filled_col = HexColor(f'#{int(scol.red*255):02x}{int(scol.green*255):02x}{int(scol.blue*255):02x}')
            c.setFillColor(filled_col)
            radius = max(1, 6 * (prob/100)**0.5)
            c.circle(sx, sy, radius, fill=1, stroke=0)
            # pct label
            c.setFillColor(GRAY)
            c.setFont('Helvetica', 5)
            c.drawCentredString(sx, sy - 10, f'{prob:.0f}%')

        # Win %
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        c.drawRightString(cols['pct'], y_cur - 12, f'{p_win}%')

        y_cur -= row_h + 2

    y_cur -= 10

    # ── DIVIDER ────────────────────────────────────────────
    c.setStrokeColor(HexColor('#E2E8F0'))
    c.line(22*mm, y_cur, W - 22*mm, y_cur)
    y_cur -= 16

    # ── SECTION : MÉTHODOLOGIE ─────────────────────────────
    c.setFillColor(GOLD)
    c.setFont('Helvetica-Bold', 7)
    c.drawString(22*mm, y_cur, 'SOUS LE CAPOT')
    y_cur -= 14
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 18)
    c.drawString(22*mm, y_cur, 'Comment fonctionne le modèle ?')
    y_cur -= 20

    # 4 method cards in 2x2 grid
    card_w = (W - 44*mm - 6) / 2
    card_h = 68
    cards = [
        ('01', 'Glicko-2 Conservative',
         'Glicko-2 · μ−σ',
         HexColor('#F59E0B'),
         'On utilise μ−σ plutôt que\nle rating brut. L\'incertitude\npénalise les équipes peu\nexposées (ex: évite Sénégal\ndevant France). Calibré sur\n378 000 matchs.'),
        ('02', 'Poisson Calibré',
         'MLE · 6 156 matchs',
         HexColor('#3B82F6'),
         'Les buts suivent une loi de\nPoisson. λ estimé par MLE\nsur données 2019-2026.\nRésultat : 1,27 buts par\néquipe dans un match\nparfaitement équilibré.'),
        ('03', 'Correction Dixon-Coles',
         'ρ = −0,011',
         HexColor('#10B981'),
         'Poisson naïf sous-estime\nles matchs nuls serrés\n(0-0, 1-1). Dixon-Coles\n(1997) corrige ce biais.\nOn obtient une matrice\ncomplète de tous les scores.'),
        ('04', 'Ancrage Pinnacle',
         'Blend modèle × marché',
         HexColor('#8B5CF6'),
         'Un modèle reste un modèle.\nOn blende nos probas avec\nles cotes Pinnacle (bookmaker\nde référence pour la\nprécision). Évite de\nsurinterpréter le modèle.'),
    ]
    for idx, (num, title, sub, accent, body) in enumerate(cards):
        col_i = idx % 2
        row_i = idx // 2
        cx = 22*mm + col_i * (card_w + 6)
        cy = y_cur - row_i * (card_h + 6)

        rect(c, cx, cy - card_h, card_w, card_h, fill=WHITE, radius=6)
        # Top accent bar
        rect(c, cx, cy - 3, card_w, 3, fill=accent, radius=2)

        # Step number (faded bg)
        c.setFillColor(HexColor('#F1F5F9'))
        c.setFont('Helvetica-Bold', 28)
        c.drawRightString(cx + card_w - 5, cy - card_h + 5, num)

        # Title
        c.setFillColor(SLATE)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(cx + 6, cy - 16, title)
        c.setFillColor(accent)
        c.setFont('Helvetica-Bold', 6)
        c.drawString(cx + 6, cy - 26, sub)

        # Body
        c.setFillColor(GRAY)
        c.setFont('Helvetica', 7)
        for li, line in enumerate(body.split('\n')):
            c.drawString(cx + 6, cy - 37 - li*8, line)

    y_cur -= 2*(card_h + 6) + 16

    # ── FOOTER ─────────────────────────────────────────────
    rect(c, 0, 0, W, 22, fill=NAVY)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 8)
    c.drawString(22*mm, 8, 'Arnaud Cordier · Projet open source')
    c.setFillColor(GOLD)
    c.drawRightString(W - 22*mm, 8, 'github.com/Cordierarn/cdm-wina')


# ═══════════════════════════════════════════════════════════
#  PAGE 2
# ═══════════════════════════════════════════════════════════

def draw_page2(c):
    rect(c, 0, 0, W, H, fill=OFFWH)

    # Header (smaller)
    hh = 52
    rect(c, 0, H - hh, W, hh, fill=NAVY)
    c.setFillColor(GOLD)
    c.setFont('Helvetica-Bold', 20)
    c.drawString(22*mm, H - 28, 'PHASE DE GROUPES')
    c.setFillColor(HexColor('#64748B'))
    c.setFont('Helvetica', 8)
    c.drawString(22*mm, H - 42, 'PROBABILITÉ DE QUALIFICATION POUR CHAQUE ÉQUIPE (50 000 SIMULATIONS)')

    y_cur = H - hh - 14

    # 12 groups in 3 columns × 4 rows
    grp_w = (W - 44*mm - 12) / 3
    grp_h = (y_cur - 28) / 4 - 6

    for gi, (letter, teams) in enumerate(GROUPS_SHORT):
        col_i = gi % 3
        row_i = gi // 3
        gx = 22*mm + col_i * (grp_w + 6)
        gy = y_cur - row_i * (grp_h + 6)

        # Card background
        rect(c, gx, gy - grp_h, grp_w, grp_h, fill=WHITE, radius=5)

        # Group header
        rect(c, gx, gy - 18, grp_w, 18, fill=NAVY, radius=5)
        # fix bottom radius
        rect(c, gx, gy - 18, grp_w, 9, fill=NAVY)

        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(gx + 5, gy - 13, f'G{letter}')
        c.setFillColor(HexColor('#94A3B8'))
        c.setFont('Helvetica', 6)
        top2 = ' · '.join(t[0][:9] for t in sorted(teams, key=lambda x:-x[2])[:2])
        c.drawString(gx + 26, gy - 13, top2)

        # Teams
        team_row_h = (grp_h - 20) / 4
        for ti, (tname, tflag, pq) in enumerate(sorted(teams, key=lambda x: -x[2])):
            ty = gy - 22 - ti * team_row_h

            # Name
            c.setFillColor(SLATE)
            c.setFont('Helvetica', 7)
            c.drawString(gx + 5, ty, tname[:12])

            # Bar
            bar_x = gx + 54
            bar_w_max = grp_w - 74
            bar_fill_w = bar_w_max * pq / 100
            bar_col = GREEN if pq >= 75 else (GOLD if pq >= 45 else LGRAY)

            rect(c, bar_x, ty - 1, bar_w_max, 5, fill=HexColor('#F1F5F9'), radius=2)
            if bar_fill_w > 2:
                rect(c, bar_x, ty - 1, bar_fill_w, 5, fill=bar_col, radius=2)

            # Pct
            c.setFillColor(GREEN if pq >= 75 else (GOLD if pq >= 45 else GRAY))
            c.setFont('Helvetica-Bold', 7)
            c.drawRightString(gx + grp_w - 3, ty, f'{pq:.0f}%')

    # ── Légende ────────────────────────────────────────────
    ly = 28
    rect(c, 0, 0, W, ly, fill=NAVY)
    c.setFillColor(WHITE)
    c.setFont('Helvetica-Bold', 7)
    c.drawString(22*mm, 17, 'Légende — probabilité de qualification :')

    items = [(GREEN,'≥75% — Favori'),(GOLD,'45–74% — Compétitif'),(LGRAY,'<45% — Outsider')]
    lx = 100*mm
    for col, lbl in items:
        c.setFillColor(col)
        c.circle(lx, 19, 4, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont('Helvetica', 7)
        c.drawString(lx + 8, 16, lbl)
        lx += 60

    c.setFillColor(GOLD)
    c.setFont('Helvetica-Bold', 7)
    c.drawRightString(W - 22*mm, 9, '#DataScience  #Football  #Statistiques')
    c.setFillColor(HexColor('#64748B'))
    c.drawRightString(W - 22*mm, 19, 'Arnaud Cordier · github.com/Cordierarn/cdm-wina')


# ═══════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════

output = str(Path(__file__).parent / 'rapport_cdm2026.pdf')
c = canvas.Canvas(output, pagesize=A4)

draw_page1(c)
c.showPage()
draw_page2(c)
c.save()

print(f'✓ PDF généré : {output}')
