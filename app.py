import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math
import os
DEFAULT_DIR = "./images/"
# -----------------------------------------------------------------------------------------------------------
# HARROUCHE BASMA 
# -----------------------------------------------------------------------------------------------------------


# -----------------------
# Fonctions utilitaires
# -----------------------

def np_to_pil(img_np):
    if img_np is None:
        return None
    return Image.fromarray(img_np.astype(np.uint8), mode='L')

def load_image_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def show_image_in_label(pil_img, label, maxsize=(480, 360)):
    if pil_img is None:
        label.configure(image=None)
        label.image = None
        return
    w, h = pil_img.size
    mw, mh = maxsize
    scale = min(1.0, mw / w, mh / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    im_resized = pil_img.resize(new_size)
    tkimg = ImageTk.PhotoImage(im_resized)
    label.configure(image=tkimg)
    label.image = tkimg

def convolution_2d(img, kernel):
    H, W = img.shape
    k = kernel.shape[0]
    pad = k // 2

    img_pad = np.pad(img, pad, mode='edge')
    res = np.zeros_like(img, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            window = img_pad[i:i + k, j:j + k]
            val = np.sum(window.astype(np.float32) * kernel)
            res[i, j] = int(round(val))
    return res

def calculer_psnr(img1, img2):
    """
    Calcule le PSNR (Peak Signal to Noise Ratio) entre deux images 8 bits.
    PSNR en dB Si les images sont identiques retourne +inf
    PSNR( I 1 , I 2)=10 log10 (255)^2 \EQM 1 2 ( ( I , I ) )
    EQM ( I 1 , I 2 )= (1\M× N) ∑ ∑( I 1 ( x , y)−I 2 ( x , y))^2
    """
    if img1 is None or img2 is None:
        return None



    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Dimensions
    M, N = img1.shape

    # EQM = erreur quadratique moyenne
    eqm = np.sum((img1 - img2) ** 2) / (M * N)

    if eqm == 0:
        return float('inf')  # images identiques

    psnr = 10 * math.log10((255.0 ** 2) / eqm)
    return psnr

# -----------------------
# Bruit
# ----------------

def bruit_additif_cv(img_cv, add_value=0):
    """Ajoute une valeur additive à tous les pixels (clip 0..255)."""
    arr = img_cv.astype(np.int32) + int(add_value)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def bruit_poivre_sel_cv(img_cv, pourcentage=10):
    """
       Poivre sel
         exmpl si pourcentage = 10 -> modifie 10% des pixels: 5% noir (0), 5% blanc (255)
         choix des pixels alatoire sans reptition
       """
    if pourcentage <= 0:
        return img_cv.copy()
    h, w = img_cv.shape
    total_pixels = h * w
    nb = int(round((pourcentage / 100.0) * total_pixels))
    nb = min(nb, total_pixels)
    noisy = img_cv.copy()

    indices = np.arange(total_pixels)
    chosen = np.random.choice(indices, size=nb, replace=False)
    half = nb // 2

    xs = chosen // w
    ys = chosen % w

    for i in range(nb):
        x = int(xs[i])
        y = int(ys[i])
        noisy[x, y] = 0 if i < half else 255

    return noisy

# -------------
# Filtres
# -------------

def filtre_moyenneur_cv(img_cv, ksize=3):
    """Moyenneur moyenne des valeurs dans la fenetre ksize x ksize"""
    if ksize <= 1:
        return img_cv.copy()
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= (ksize * ksize)
    return convolution_2d(img_cv, kernel)


def filtre_gaussien_cv(img_cv, sigma=1.0):
    """Gaussien calcule masque de taille voisinage = 6*sigma (arrondi à impair) puis convolution"""
    if sigma <= 0:
        return img_cv.copy()
    size = int(round(6 * sigma))
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    center = (size - 1) / 2.0
    ax = np.arange(size) - center
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * (sigma ** 2)))
    kernel = kernel / np.sum(kernel)
    return convolution_2d(img_cv, kernel)



def filtre_median_cv(img_cv, ksize=3):
    """Filtre median valeur mediane dans la fenetre de voisinage ksize x ksize"""
    if ksize <= 1:
        return img_cv.copy()
    pad = ksize // 2
    img_pad = np.pad(img_cv, pad, mode='edge')
    H, W = img_cv.shape
    res = np.zeros_like(img_cv, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            window = img_pad[i:i + ksize, j:j + ksize].flatten()
            res[i, j] = int(np.median(window))

    return res

def filtre_min_max_cv(img_cv, ksize=3):
    """
       min max
         pour chaque pixel on recupere toutes les valeurs dans lee voisinage
         si pixel central < min -> remplacer par min
          elif pixel central > max -> remplacer par max
          else -> laisser inchanger
       """
    if ksize <= 1:
        return img_cv.copy()
    pad = ksize // 2
    img_pad = np.pad(img_cv, pad, mode='edge')
    H, W = img_cv.shape
    res = np.zeros_like(img_cv, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            window = img_pad[i:i + ksize, j:j + ksize].flatten()
            wmin = int(np.min(window))
            wmax = int(np.max(window))
            center_val = int(img_cv[i, j])

            if center_val < wmin:
                res[i, j] = wmin
            elif center_val > wmax:
                res[i, j] = wmax
            else:
                res[i, j] = center_val

    return res

# ---------------
# Nom automatique
# ---------------

def generate_auto_filename(operation_name, extension=".png"):
    base = operation_name.lower().replace(" ", "_")
    i = 1
    while True:
        filename = f"{base}_{i}{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

# -------------------
# Feneetre Operation
# --------------------

class OperationWindow(tk.Toplevel):
    def __init__(self, master, title):
        super().__init__(master)
        self.title(title)
        self.img_orig = None
        self.img_result = None

        tk.Label(self, text="Image originale").pack()
        self.canvas_orig = tk.Label(self)
        self.canvas_orig.pack(padx=6, pady=6)

        tk.Label(self, text="Résultat").pack()
        self.canvas_res = tk.Label(self)
        self.canvas_res.pack(padx=6, pady=6)

        frame = tk.Frame(self)
        frame.pack(pady=6)
        tk.Button(frame, text="Choisir image", command=self.choisir_image).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="Appliquer", command=self.appliquer_action).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="Calculer PSNR", command=self.calculer_psnr_btn).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="Sauvegarder résultat", command=self.sauver_resultat).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="Fermer", command=self.destroy).pack(side=tk.LEFT, padx=4)

        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(pady=6)

    def choisir_image(self):
        path = filedialog.askopenfilename(
            initialdir=DEFAULT_DIR,
            filetypes=[("Images", "*.bmp;*.png;*.jpg;*.jpeg")]
        )
        if not path:
            return
        self.img_orig = load_image_gray(path)
        show_image_in_label(np_to_pil(self.img_orig), self.canvas_orig)
        self.img_result = None
        self.canvas_res.configure(image=None)
        self.canvas_res.image = None

    def appliquer_action(self):
        raise NotImplementedError

    def calculer_psnr_btn(self):
        if self.img_orig is None or self.img_result is None:
            messagebox.showwarning("Alerte", "Image originale ou résultat manquant.")
            return
        psnr = calculer_psnr(self.img_orig, self.img_result)
        if psnr == float('inf'):
            messagebox.showinfo("PSNR", "PSNR = inf (images identiques)")
        else:
            messagebox.showinfo("PSNR", f"PSNR = {psnr:.2f} dB")

    def sauver_resultat(self):
        if self.img_result is None:
            messagebox.showwarning("Alerte", "Aucun résultat à sauvegarder.")
            return

        operation_name = self.title()
        auto_name = generate_auto_filename(operation_name, ".bmp")

        path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_DIR,
            initialfile=auto_name,
            defaultextension=".bmp",
            filetypes=[("bmp image", "*.bmp")]
        )
        if not path:
            return

        pil = np_to_pil(self.img_result)
        pil.save(path)
        messagebox.showinfo("Sauvegardé", f"Image enregistrée : {path}")

# ---------------------
# Fenetres Specifiques
# ---------------------

class BruitAdditifWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Bruit Additif")
        tk.Label(self.controls_frame, text="Valeur additive (0..255):").pack(side=tk.LEFT)
        self.add_var = tk.IntVar(value=0)
        tk.Spinbox(self.controls_frame, from_=0, to=255, textvariable=self.add_var, width=6).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        add = int(self.add_var.get())
        self.img_result = bruit_additif_cv(self.img_orig, add_value=add)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


class BruitPoivreSelWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Bruit Poivre & Sel")
        tk.Label(self.controls_frame, text="Pourcentage (%):").pack(side=tk.LEFT)
        self.p_var = tk.IntVar(value=10)
        tk.Spinbox(self.controls_frame, from_=0, to=100, textvariable=self.p_var, width=6).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        p = int(self.p_var.get())
        self.img_result = bruit_poivre_sel_cv(self.img_orig, pourcentage=p)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


class FiltreMoyenneurWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Filtre Moyenneur")
        tk.Label(self.controls_frame, text="Voisinage (impair):").pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(self.controls_frame, from_=3, to=31, increment=2, textvariable=self.k_var, width=4).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        k = int(self.k_var.get())
        self.img_result = filtre_moyenneur_cv(self.img_orig, ksize=k)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


class FiltreGaussienWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Filtre Gaussien")
        tk.Label(self.controls_frame, text="Sigma > 0 :").pack(side=tk.LEFT)
        self.s_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(self.controls_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.s_var, width=6).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        sigma = float(self.s_var.get())
        self.img_result = filtre_gaussien_cv(self.img_orig, sigma=sigma)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


class FiltreMedianWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Filtre Médian")
        tk.Label(self.controls_frame, text="Voisinage (impair):").pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(self.controls_frame, from_=3, to=31, increment=2, textvariable=self.k_var, width=4).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        k = int(self.k_var.get())
        self.img_result = filtre_median_cv(self.img_orig, ksize=k)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


class FiltreMinMaxWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Filtre Min-Max")
        tk.Label(self.controls_frame, text="Voisinage (impair):").pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(self.controls_frame, from_=3, to=31, increment=2, textvariable=self.k_var, width=4).pack(side=tk.LEFT)

    def appliquer_action(self):
        if self.img_orig is None:
            messagebox.showwarning("Alerte", "Choisir d'abord une image.")
            return
        k = int(self.k_var.get())
        self.img_result = filtre_min_max_cv(self.img_orig, ksize=k)
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


# -----------------------
# Application principale
# -----------------------

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bruit & Filtrage")
        self.geometry("560x360")

        tk.Label(self, text="Application : Bruit et Filtrage", font=("Arial", 14)).pack(pady=8)

        frame = tk.Frame(self)
        frame.pack(pady=10)

        tk.Button(frame, text="Bruit Additif", width=24,
                  command=lambda: BruitAdditifWindow(self)).grid(row=0, column=0, padx=6, pady=6)

        tk.Button(frame, text="Bruit Poivre & Sel", width=24,
                  command=lambda: BruitPoivreSelWindow(self)).grid(row=0, column=1, padx=6, pady=6)

        tk.Button(frame, text="Filtre Moyenneur", width=24,
                  command=lambda: FiltreMoyenneurWindow(self)).grid(row=1, column=0, padx=6, pady=6)

        tk.Button(frame, text="Filtre Gaussien", width=24,
                  command=lambda: FiltreGaussienWindow(self)).grid(row=1, column=1, padx=6, pady=6)

        tk.Button(frame, text="Filtre Médian", width=24,
                  command=lambda: FiltreMedianWindow(self)).grid(row=2, column=0, padx=6, pady=6)

        tk.Button(frame, text="Filtre Min-Max", width=24,
                  command=lambda: FiltreMinMaxWindow(self)).grid(row=2, column=1, padx=6, pady=6)

        tk.Label(self, text="Choisir un bruit / filtre.", wraplength=520).pack(pady=6)


# ------------
# Lancer l'application
# ------------

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
