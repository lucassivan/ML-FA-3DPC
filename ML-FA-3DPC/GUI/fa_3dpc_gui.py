"""
FA-3DPC Strength Predictor
Models: ExtraTrees (CS) · ElasticNet (FS)  |  LOMO CV  |  n=126
"""
import os
import threading
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# ── model constants ───────────────────────────────────────────────────────────
FEATURES  = ['FA_pct', 'W_B', 'Age', 'ln_Age', 'FA_Age']
ET_PARAMS = dict(n_estimators=205, max_depth=7, min_samples_split=4,
                 min_samples_leaf=1, max_features=None, random_state=42, n_jobs=-1)
EN_PARAMS = dict(alpha=0.0075, l1_ratio=0.951, max_iter=10000)
PI90_CS   = 4.595
PI90_FS   = 0.494

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    'header':    '#1C2B4A',
    'g_binder':  '#1A3A5C',
    'g_mix':     '#1B5E20',
    'g_cure':    '#4E342E',
    'cs':        '#C62828',
    'fs':        '#6D4C41',
    'btn_p':     '#1C2B4A',   'btn_ph': '#263350',
    'btn_e':     '#E65100',   'btn_eh': '#BF360C',
    'btn_c':     '#546E7A',   'btn_ch': '#37474F',
    'bg':        '#F5F6FA',
    'card':      '#FFFFFF',
    'border':    '#DDE1E9',
    'secondary': '#4A6572',   # secondary field labels — readable dark slate
    'unit':      '#546E7A',   # unit / annotation text
    'auto_bg':   '#EEF1F5',
    'auto_txt':  '#546E7A',   # auto-field values — same as unit for consistency
    'annot':     '#607D8B',   # "auto" / "fixed" badges
    'idle':      '#90A4AE',
}

EXAMPLE   = {'FA_pct': '7.5', 'W_B': '0.30', 'Age': '28'}
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'dataset.csv')


# ── data & training ───────────────────────────────────────────────────────────
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Sand (g)', 'Accelerator (%)', 'SP (%)'])
    df = df.rename(columns={
        'Fly ash (g)': 'FA_pct',
        'W/B':         'W_B',
        'Age (days)':  'Age',
    })
    df['ln_Age'] = np.log(df['Age'])
    df['FA_Age'] = df['FA_pct'] * df['Age']
    X, y_cs, y_fs = (df[FEATURES].values,
                     df['CS (MPa)'].values,
                     df['FS (MPa)'].values)
    et = ExtraTreesRegressor(**ET_PARAMS);  et.fit(X, y_cs)
    sc = StandardScaler().fit(X)
    en = ElasticNet(**EN_PARAMS);           en.fit(sc.transform(X), y_fs)
    return et, en, sc


# ── grade helpers ─────────────────────────────────────────────────────────────
def _cs_grade(v):
    if v < 35:  return "Low compressive strength"
    if v < 45:  return "Moderate compressive strength"
    if v < 55:  return "High compressive strength"
    return "Very high compressive strength"

def _fs_grade(v):
    if v < 4.5: return "Low flexural strength"
    if v < 6.5: return "Moderate flexural strength"
    if v < 8.0: return "High flexural strength"
    return "Very high flexural strength"


# ── application ───────────────────────────────────────────────────────────────
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("FA-3DPC Strength Predictor")
        self.geometry("1080x720")
        self.minsize(960, 660)
        self.configure(fg_color=C['bg'])
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self._et = self._en = self._sc = None
        self._ready = False
        self._build()
        threading.Thread(target=self._train_bg, daemon=True).start()

    # ── background training ───────────────────────────────────────────────────
    def _train_bg(self):
        self._status("Training models on dataset…")
        try:
            et, en, sc = load_and_train()
            self._et, self._en, self._sc = et, en, sc
            self._ready = True
            self.after(0, lambda: self._status("Models ready."))
        except Exception as ex:
            self.after(0, lambda: self._status(f"Error: {ex}"))

    # ── UI construction ───────────────────────────────────────────────────────
    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_banner()
        self._build_body()

    # ── banner ────────────────────────────────────────────────────────────────
    def _build_banner(self):
        bn = ctk.CTkFrame(self, fg_color=C['header'], corner_radius=0, height=78)
        bn.grid(row=0, column=0, sticky='ew')
        bn.grid_propagate(False)
        bn.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bn,
                     text="FA-3DPC Strength Predictor",
                     font=ctk.CTkFont("Arial", 25, "bold"),
                     text_color="#FFFFFF").grid(row=0, column=0, pady=(16, 3))
        ctk.CTkLabel(bn,
                     text="Fly Ash Blended 3D-Printed Concrete  ·  "
                          "CS & FS Prediction with 90% Prediction Intervals",
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#90A4AE").grid(row=1, column=0, pady=(0, 10))

    # ── body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=C['bg'], corner_radius=0)
        body.grid(row=1, column=0, sticky='nsew', padx=18, pady=14)
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self._build_input_card(body)
        self._build_output_row(body)

    # ── input card ────────────────────────────────────────────────────────────
    def _build_input_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        card.grid_columnconfigure(0, weight=3)
        card.grid_columnconfigure(1, weight=0)
        card.grid_columnconfigure(2, weight=4)
        card.grid_columnconfigure(3, weight=0)
        card.grid_columnconfigure(4, weight=2)

        self._build_group_binder(card, col=0)

        ctk.CTkFrame(card, fg_color=C['border'], width=1).grid(
            row=0, column=1, sticky='ns', padx=2, pady=14)

        self._build_group_mix(card, col=2)

        ctk.CTkFrame(card, fg_color=C['border'], width=1).grid(
            row=0, column=3, sticky='ns', padx=2, pady=14)

        self._build_group_curing(card, col=4)

        # button row
        btn_frame = ctk.CTkFrame(card, fg_color='transparent')
        btn_frame.grid(row=1, column=0, columnspan=5, pady=(2, 16))

        ctk.CTkButton(btn_frame, text="PREDICT", width=140, height=46,
                      font=ctk.CTkFont("Arial", 14, "bold"),
                      fg_color=C['btn_p'], hover_color=C['btn_ph'],
                      corner_radius=8, command=self._predict
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text="EXAMPLE", width=115, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_e'], hover_color=C['btn_eh'],
                      corner_radius=8, command=self._example
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text="CLEAR", width=105, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_c'], hover_color=C['btn_ch'],
                      corner_radius=8, command=self._clear
                      ).pack(side='left', padx=8)

        ctk.CTkLabel(btn_frame,
                     text="  * active model inputs     auto = derived from inputs"
                          "     fixed = constant in dataset",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).pack(side='left', padx=14)

    # ── group: Binder Composition ─────────────────────────────────────────────
    def _build_group_binder(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)

        self._group_hdr(grp, "BINDER COMPOSITION", C['g_binder'])

        # FA% — active input
        ctk.CTkLabel(grp, text="FA Content *",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color=C['g_binder']).grid(
            row=1, column=0, sticky='e', padx=(0, 8), pady=7)
        self._e_fa = ctk.CTkEntry(grp, width=88, height=36,
                                   font=ctk.CTkFont("Arial", 13),
                                   corner_radius=6, border_color=C['border'])
        self._e_fa.grid(row=1, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="wt.%",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['unit']).grid(row=1, column=2, sticky='w', padx=(4, 0))

        # Cement — auto
        ctk.CTkLabel(grp, text="Cement",
                     font=ctk.CTkFont("Arial", 13),
                     text_color=C['secondary']).grid(
            row=2, column=0, sticky='e', padx=(0, 8), pady=7)
        self._lbl_cement = self._auto_field(grp, "—", width=88)
        self._lbl_cement.grid(row=2, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="wt.%  auto",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).grid(row=2, column=2, sticky='w', padx=(4, 0))

        self._e_fa.bind("<KeyRelease>", self._update_derived)

    # ── group: Mix Proportions ────────────────────────────────────────────────
    def _build_group_mix(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)

        self._group_hdr(grp, "MIX PROPORTIONS", C['g_mix'])

        # W/B — active input
        ctk.CTkLabel(grp, text="W/B Ratio *",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color=C['g_mix']).grid(
            row=1, column=0, sticky='e', padx=(0, 8), pady=7)
        self._e_wb = ctk.CTkEntry(grp, width=88, height=36,
                                   font=ctk.CTkFont("Arial", 13),
                                   corner_radius=6, border_color=C['border'])
        self._e_wb.grid(row=1, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="—",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['unit']).grid(row=1, column=2, sticky='w', padx=(4, 0))

        # SP — auto
        ctk.CTkLabel(grp, text="SP Dosage",
                     font=ctk.CTkFont("Arial", 13),
                     text_color=C['secondary']).grid(
            row=2, column=0, sticky='e', padx=(0, 8), pady=7)
        self._lbl_sp = self._auto_field(grp, "—", width=88)
        self._lbl_sp.grid(row=2, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="wt.%  auto",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).grid(row=2, column=2, sticky='w', padx=(4, 0))

        # Sand — fixed
        ctk.CTkLabel(grp, text="Sand",
                     font=ctk.CTkFont("Arial", 13),
                     text_color=C['secondary']).grid(
            row=3, column=0, sticky='e', padx=(0, 8), pady=7)
        self._auto_field(grp, "1000", width=88).grid(row=3, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="g  fixed",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).grid(row=3, column=2, sticky='w', padx=(4, 0))

        # Accelerator — fixed
        ctk.CTkLabel(grp, text="Accelerator",
                     font=ctk.CTkFont("Arial", 13),
                     text_color=C['secondary']).grid(
            row=4, column=0, sticky='e', padx=(0, 8), pady=7)
        self._auto_field(grp, "1.4", width=88).grid(row=4, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="wt.%  fixed",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).grid(row=4, column=2, sticky='w', padx=(4, 0))

        self._e_wb.bind("<KeyRelease>", self._update_derived)

    # ── group: Curing ─────────────────────────────────────────────────────────
    def _build_group_curing(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)

        self._group_hdr(grp, "CURING", C['g_cure'])

        ctk.CTkLabel(grp, text="Age *",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color=C['g_cure']).grid(
            row=1, column=0, sticky='e', padx=(0, 8), pady=7)
        self._e_age = ctk.CTkEntry(grp, width=88, height=36,
                                    font=ctk.CTkFont("Arial", 13),
                                    corner_radius=6, border_color=C['border'])
        self._e_age.grid(row=1, column=1, padx=4, pady=7)
        ctk.CTkLabel(grp, text="days",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['unit']).grid(row=1, column=2, sticky='w', padx=(4, 0))

        ctk.CTkLabel(grp, text="Valid range: 1 – 28 d",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).grid(
            row=2, column=0, columnspan=3, pady=(2, 0))

    # ── output row ────────────────────────────────────────────────────────────
    def _build_output_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color='transparent')
        row.grid(row=1, column=0, sticky='nsew')
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        row.grid_rowconfigure(0, weight=1)

        for col, tag, title, color in (
            (0, 'cs', 'Compressive Strength', C['cs']),
            (1, 'fs', 'Flexural Strength',    C['fs']),
        ):
            card = ctk.CTkFrame(row, fg_color=C['card'], corner_radius=12,
                                border_width=1, border_color=C['border'])
            card.grid(row=0, column=col, sticky='nsew',
                      padx=(0, 10) if col == 0 else (10, 0))
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)

            hdr = ctk.CTkFrame(card, fg_color=color, corner_radius=8, height=40)
            hdr.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 10))
            hdr.grid_propagate(False)
            ctk.CTkLabel(hdr, text=title,
                         font=ctk.CTkFont("Arial", 14, "bold"),
                         text_color="white").place(relx=0.5, rely=0.5, anchor='center')

            inner = ctk.CTkFrame(card, fg_color='transparent')
            inner.grid(row=1, column=0, sticky='nsew')
            inner.grid_columnconfigure(0, weight=1)
            inner.grid_rowconfigure(0, weight=1)

            lbl_val = ctk.CTkLabel(inner, text="— MPa",
                                   font=ctk.CTkFont("Arial", 42, "bold"),
                                   text_color=color)
            lbl_val.grid(row=0, column=0, pady=(16, 8))

            lbl_pi = ctk.CTkLabel(inner, text="90% PI:  —",
                                  font=ctk.CTkFont("Arial", 14),
                                  text_color=C['secondary'])
            lbl_pi.grid(row=1, column=0, pady=(0, 8))

            lbl_grade = ctk.CTkLabel(inner, text="",
                                     font=ctk.CTkFont("Arial", 13, slant="italic"),
                                     text_color=C['idle'])
            lbl_grade.grid(row=2, column=0, pady=(0, 20))

            setattr(self, f'_lbl_{tag}_val',   lbl_val)
            setattr(self, f'_lbl_{tag}_pi',    lbl_pi)
            setattr(self, f'_lbl_{tag}_grade', lbl_grade)


    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _group_hdr(parent, text, color):
        hdr = ctk.CTkFrame(parent, fg_color=color, corner_radius=6, height=30)
        hdr.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 8))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text=text,
                     font=ctk.CTkFont("Arial", 11, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

    @staticmethod
    def _auto_field(parent, text, width=88):
        frm = ctk.CTkFrame(parent, fg_color=C['auto_bg'], corner_radius=6,
                           border_width=1, border_color=C['border'],
                           width=width, height=36)
        frm.grid_propagate(False)
        lbl = ctk.CTkLabel(frm, text=text,
                           font=ctk.CTkFont("Arial", 13),
                           text_color=C['auto_txt'])
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        frm._inner_label = lbl
        return frm

    # ── auto-update derived fields ────────────────────────────────────────────
    def _update_derived(self, _event=None):
        try:
            fa = float(self._e_fa.get())
            self._lbl_cement._inner_label.configure(text=f"{max(0.0, 100.0 - fa):.1f}")
        except ValueError:
            self._lbl_cement._inner_label.configure(text="—")

        try:
            wb = float(self._e_wb.get())
            sp = 0.30 if wb <= 0.31 else 0.00
            self._lbl_sp._inner_label.configure(text=f"{sp:.2f}")
        except ValueError:
            self._lbl_sp._inner_label.configure(text="—")

    # ── predict ───────────────────────────────────────────────────────────────
    def _predict(self):
        if not self._ready:
            messagebox.showwarning("Not Ready", "Models are still loading.")
            return
        try:
            fa  = float(self._e_fa.get())
            wb  = float(self._e_wb.get())
            age = float(self._e_age.get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Please enter numeric values for FA%, W/B, and Age.")
            return

        warns = []
        if not 0 <= fa <= 15:       warns.append("FA Content outside training range (0–15 wt.%)")
        if not 0.28 <= wb <= 0.37:  warns.append("W/B outside training range (0.30–0.35)")
        if not 0 < age <= 90:       warns.append("Age outside training range (1–28 days)")
        if warns:
            if not messagebox.askyesno(
                    "Out of Range",
                    "\n".join(warns) + "\n\nProceed with extrapolated prediction?"):
                return

        X  = np.array([[fa, wb, age, np.log(age), fa * age]])
        cs = float(self._et.predict(X)[0])
        fs = float(self._en.predict(self._sc.transform(X))[0])

        self._lbl_cs_val.configure(text=f"{cs:.1f} MPa")
        self._lbl_cs_pi.configure(
            text=f"90% PI:  [{cs - PI90_CS:.1f},  {cs + PI90_CS:.1f}] MPa")
        self._lbl_cs_grade.configure(text=_cs_grade(cs))

        self._lbl_fs_val.configure(text=f"{fs:.2f} MPa")
        self._lbl_fs_pi.configure(
            text=f"90% PI:  [{fs - PI90_FS:.3f},  {fs + PI90_FS:.3f}] MPa")
        self._lbl_fs_grade.configure(text=_fs_grade(fs))

        self._update_derived()
        self._status(
            f"Predicted  —  FA = {fa} wt.%    W/B = {wb}    Age = {age:.0f} d")

    # ── example / clear ───────────────────────────────────────────────────────
    def _example(self):
        self._clear()
        self._e_fa.insert(0,  EXAMPLE['FA_pct'])
        self._e_wb.insert(0,  EXAMPLE['W_B'])
        self._e_age.insert(0, EXAMPLE['Age'])
        self._update_derived()
        self._status("Example values loaded (FA=7.5%, W/B=0.30, Age=28 d).")

    def _clear(self):
        for w in (self._e_fa, self._e_wb, self._e_age):
            w.delete(0, 'end')
        self._lbl_cement._inner_label.configure(text="—")
        self._lbl_sp._inner_label.configure(text="—")
        for tag in ('cs', 'fs'):
            getattr(self, f'_lbl_{tag}_val').configure(text="— MPa")
            getattr(self, f'_lbl_{tag}_pi').configure(text="90% PI:  —")
            getattr(self, f'_lbl_{tag}_grade').configure(text="")
        self._status("Cleared.")

    def _status(self, msg: str):
        pass   # status bar removed


if __name__ == '__main__':
    app = App()
    app.mainloop()
