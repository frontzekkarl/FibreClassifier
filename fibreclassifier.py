import numpy as np
import tifffile
import pandas as pd
import matplotlib
if __name__ == "__main__":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.errors import TopologicalError
from skimage.draw import polygon as sk_polygon
from sklearn.cluster import KMeans
import os
import json

# At the very top of your script
root = tk.Tk()
root.withdraw()
def select_file(title, filetypes):
    filename = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=root)
    if not filename:
        raise ValueError(f"No file selected for {title}.")
    return filename

def ask_save_file(title, default_filename):
    file_path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".csv",
        initialfile=default_filename,
        filetypes=[("CSV files", "*.csv")],
        parent=root
    )
    return file_path
# --- Step 1: Select image and outline file ---
print("Select input files...")
image_file = select_file("Select 8-bit TIFF Image", [("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
outline_file = select_file("Select Fibre Outline TXT File", [("Text files", "*.txt"), ("All files", "*.*")])
outline_cleaned_file = outline_file.replace(".txt", "_cleaned.txt")

# Load outlines (cleaned if available and chosen)
if os.path.exists(outline_cleaned_file):
    use_cleaned = messagebox.askyesno(
        "Use Cleaned Outline?",
        f"A cleaned outline was found:\n{os.path.basename(outline_cleaned_file)}\nUse it instead?"
    )
    with open(outline_cleaned_file if use_cleaned else outline_file, 'r') as f:
        lines = f.readlines()
else:
    with open(outline_file, 'r') as f:
        lines = f.readlines()

# Load image
image = tifffile.imread(image_file)
if image.ndim > 2:
    image = image[0]
if image.ndim != 2:
    raise ValueError("Loaded image must be 2D.")
import imagej

ij = imagej.init(mode='headless')  # or mode='interactive' if you prefer

jimage = ij.io().open(image_file)

pixel_size_x = pixel_size_y = None
num_dims = jimage.numDimensions()
for i in range(num_dims):
    axis = jimage.axis(i)
    axis_type = axis.type().getLabel()
    scale = axis.scale()
    if axis_type == "X":
        pixel_size_x = scale
    elif axis_type == "Y":
        pixel_size_y = scale

if pixel_size_x is None or pixel_size_y is None:
    raise ValueError("Failed to extract pixel calibration from TIFF file.")

pixel_area_um2 = pixel_size_x * pixel_size_y
print(f"Pixel area (µm²): {pixel_area_um2}")

# --- Step 2: Segment fibres ---
def run_segmentation(lines, image):
    results = []
    fibre_polygons = {}
    fibre_coords = {}

    for i, line in enumerate(lines):
        coords = list(map(int, line.strip().split(',')))
        x = np.array(coords[::2])
        y = np.array(coords[1::2])

        if len(x) < 3:
            continue

        poly = Polygon(zip(x, y))
        if not poly.is_valid or poly.area == 0:
            try:
                poly = poly.buffer(0)
                print(f"Fibre {i+1}: repaired")
            except TopologicalError:
                continue

        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)

        if not poly.is_valid or poly.area == 0:
            continue

        x_poly, y_poly = poly.exterior.coords.xy
        rr, cc = sk_polygon(np.array(y_poly, dtype=int), np.array(x_poly, dtype=int), image.shape)
        rr = np.clip(rr, 0, image.shape[0] - 1)
        cc = np.clip(cc, 0, image.shape[1] - 1)
        mask = np.zeros_like(image, dtype=bool)
        mask[rr, cc] = True

        pixel_vals = image[mask]
        if pixel_vals.size == 0:
            continue
        centroid = poly.centroid
        mean_val = np.mean(pixel_vals)
        std_val = np.std(pixel_vals)
        area_px = poly.area
        convex_area_px = poly.convex_hull.area
        perimeter_px = poly.length

        area_um2 = area_px * pixel_area_um2
        convex_area_um2 = convex_area_px * pixel_area_um2
        convexity_ratio = area_um2 / convex_area_um2 if convex_area_um2 > 0 else np.nan
        roundness = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan

        fibre_id = i + 1
        results.append({
            "fibre_id": fibre_id,
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2),
            "area_um2": round(area_um2, 2),
            "n_pixels": int(pixel_vals.size),
            "convexity_ratio": round(convexity_ratio, 4),
            "roundness": round(roundness, 4),
            "centroid_x": round(centroid.x, 2),
            "centroid_y": round(centroid.y, 2)
        })

        fibre_polygons[fibre_id] = poly
        fibre_coords[fibre_id] = coords

    df = pd.DataFrame(results)
    return df, fibre_polygons, fibre_coords

def ensure_segmented():
    global df, fibre_polygons, fibre_coords, lines

    if os.path.exists("fibre_excluded_results.csv"):
        use_existing = messagebox.askyesno(
            "Use Existing Segmentation?",
            "A saved fibre_excluded_results.csv file was found.\nDo you want to load it instead of rerunning segmentation?"
        )
        if use_existing:
            df = pd.read_csv("fibre_excluded_results.csv")
            with open("fibre_coords.json", "r") as f:
                fibre_coords = json.load(f)
                fibre_coords = {int(k): v for k, v in fibre_coords.items()}
            fibre_polygons = {}
            for fid, coords in fibre_coords.items():
                x = np.array(coords[::2])
                y = np.array(coords[1::2])
                fibre_polygons[fid] = Polygon(zip(x, y))
            return

    # Otherwise rerun segmentation
    df, fibre_polygons, fibre_coords = run_segmentation(lines, image)
# --- Manual Exclusion ---
def manual_exclude():
    global df, fibre_polygons, fibre_coords
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Click Fibres to Exclude/Include, ENTER to Save", fontsize=12)
    ax.axis('off')

    fibre_texts = {}
    included_fibres = {}

    for _, row in df.iterrows():
        fid = row['fibre_id']
        poly = fibre_polygons[fid]
        x_poly, y_poly = poly.exterior.coords.xy
        ax.plot(x_poly, y_poly, color='red', linewidth=0.8)
        txt = ax.text(row['centroid_x'], row['centroid_y'], str(fid), color='yellow',
                      fontsize=6, ha='center', va='center',
                      bbox=dict(facecolor='black', alpha=0.3, pad=1))
        fibre_texts[fid] = txt
        included_fibres[fid] = True

    def on_click(event):
        if event.inaxes != ax:
            return
        point = Point(event.xdata, event.ydata)
        for fid, poly in fibre_polygons.items():
            if poly.contains(point):
                included_fibres[fid] = not included_fibres[fid]
                fibre_texts[fid].set_color('yellow' if included_fibres[fid] else 'grey')
                fig.canvas.draw()
                break
    def on_key(event):
        if event.key == 'enter':
            # Keep original fibre IDs before resetting
            df_filtered = df[df['fibre_id'].map(included_fibres)].copy()
            df_filtered['original_fibre_id'] = df_filtered['fibre_id']
            df_filtered = df_filtered.reset_index(drop=True)
            df_filtered['fibre_id'] = df_filtered.index + 1  # assign new IDs

            # Rebuild dictionary with new IDs but using original polygon mapping
            new_fibre_polygons = {}
            new_fibre_coords = {}
            for _, row in df_filtered.iterrows():
                new_id = row['fibre_id']
                old_id = int(row['original_fibre_id'])  # cast to int to match original dict keys
                new_fibre_polygons[new_id] = fibre_polygons[old_id]
                new_fibre_coords[new_id] = fibre_coords[old_id]

            df_filtered.to_csv("fibre_excluded_results.csv", index=False)
            with open("fibre_coords.json", "w") as f:
                json.dump(new_fibre_coords, f)

            cleaned_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")],
                title="Save Cleaned Outline As"
            )
            if cleaned_path:
                with open(cleaned_path, "w") as f:
                    for coords in new_fibre_coords.values():
                        f.write(",".join(map(str, coords)) + "\n")
                print("Cleaned outline and results saved.")
            else:
                print("Cleaned outline not saved.")
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
def plot_coloured_classification(df_plot, status_col, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Overlay: {status_col}", fontsize=12)
    ax.axis('off')

    color_map = {
        "positive": 'red',
        "equivocal": 'blue',
        "negative": 'green'
    }

    for _, row in df_plot.iterrows():
        fibre_id = row['fibre_id']
        poly = fibre_polygons[fibre_id]
        label = row[status_col]
        color = color_map.get(label, 'yellow')

        x_poly, y_poly = poly.exterior.coords.xy
        ax.plot(x_poly, y_poly, color=color, linewidth=1.2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved overlay: {save_path}")
        plt.close(fig)
    else:
        plt.show()
def manual_classify(starting_labels=None):
    global df, fibre_polygons, image

    label_selector = tk.Toplevel()
    label_selector.title("Select Label")
    label_selector.geometry("200x100")

    current_label = tk.StringVar(value="positive")
    label_cycle = ["positive", "equivocal", "negative"]

    tk.Label(label_selector, text="Classification Label").pack(pady=5)
    tk.OptionMenu(label_selector, current_label, *label_cycle).pack(pady=5)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Click fibres, press ENTER to save", fontsize=12)
    ax.axis("off")

    fibre_texts = {}
    manual_labels = {}

    for _, row in df.iterrows():
        fid = int(row["fibre_id"])
        poly = fibre_polygons[fid]
        x_poly, y_poly = poly.exterior.coords.xy
        ax.plot(x_poly, y_poly, color="red", linewidth=0.8)

        label = starting_labels.get(fid, "") if starting_labels else ""
        manual_labels[fid] = label
        txt = ax.text(
            row["centroid_x"], row["centroid_y"],
            f"{fid}\n{label}",
            color="yellow", fontsize=6, ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.3, pad=1)
        )
        fibre_texts[fid] = txt

    def on_click(event):
        if event.inaxes != ax:
            return
        pt = Point(event.xdata, event.ydata)
        label = current_label.get()
        for fid, poly in fibre_polygons.items():
            if poly.contains(pt):
                manual_labels[fid] = label
                fibre_texts[fid].set_text(f"{fid}\n{label}")
                fibre_texts[fid].set_color({
                    "positive": "red",
                    "equivocal": "blue",
                    "negative": "green"
                }.get(label, "yellow"))
                fig.canvas.draw()
                break

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
            label_selector.destroy()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return manual_labels
def classify_menu():
    global df, fibre_polygons, fibre_coords, lines, image

    # Ask if user wants to load a different outline
    use_other = messagebox.askyesno("Use Other Outline?", "Do you want to use a different segmentation outline TXT file?")
    if use_other:
        alt_outline = filedialog.askopenfilename(
            title="Select segmentation TXT file",
            filetypes=[("Text files", "*.txt")]
        )
        if alt_outline:
            with open(alt_outline, 'r') as f:
                lines = f.readlines()
        else:
            messagebox.showinfo("Cancelled", "No new outline selected. Using original.")

    # Run segmentation
    df, fibre_polygons, fibre_coords = run_segmentation(lines, image)

    # Ask for manual or automatic classification
    mode = messagebox.askquestion(
        "Classification Mode",
        "Do you want to classify fibres manually (paintbrush style)?\nChoose 'No' for auto-thresholding."
    )

    if mode == 'yes':
        labels = {}
        labels = manual_classify(starting_labels=labels)
        df['manual_status'] = df['fibre_id'].astype(int).map(labels)
        save = messagebox.askyesno("Save Manual Classification?", "Do you want to save the manual classification?")
        if save:
            df['manual_status'] = df['fibre_id'].astype(int).map(labels)
            df.to_csv("fibre_manual_classified.csv", index=False)
            print("Saved: fibre_manual_classified.csv")
        show_menu()
    else:
        # Automatic thresholding via KMeans
        X = df["mean_intensity"].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
        df["kmeans_cluster"] = kmeans.labels_

        cluster_means = df.groupby("kmeans_cluster")["mean_intensity"].mean()
        sorted_clusters = cluster_means.sort_values().index.tolist()

        cluster_to_status = {
            sorted_clusters[0]: "positive",
            sorted_clusters[1]: "equivocal",
            sorted_clusters[2]: "negative"
        }
        df["ihc_status"] = df["kmeans_cluster"].map(cluster_to_status)
        df.drop(columns="kmeans_cluster", inplace=True)

        df.to_csv("fibre_auto_thresholded.csv", index=False)
        print("Saved: fibre_auto_thresholded.csv")

        plot_coloured_classification(df, status_col="ihc_status", save_path="overlay_auto_thresholded.png")

        # Optionally amend
        amend = messagebox.askyesno("Amend Classification?", "Do you want to review and edit class assignments?")
        if amend:
            labels = df.set_index("fibre_id")["ihc_status"].to_dict()
            labels = manual_classify(starting_labels=labels)
            df['manual_status'] = df['fibre_id'].astype(int).map(labels)
            save = messagebox.askyesno("Save Revised Classification?", "Do you want to save the manually revised classification?")
            if save:
                df['manual_status'] = df['fibre_id'].astype(int).map(labels)
                df.to_csv("fibre_manual_classified.csv", index=False)
                print("Saved: fibre_manual_classified.csv")
                plot_coloured_classification(df, status_col="manual_status", save_path="overlay_manual_amended.png")

        show_menu()
# --- Step 7: GUI Menu ---
def show_menu():
    root = tk.Tk()
    root.title("Choose Action")

    label = tk.Label(root, text="Select Next Action", font=("Arial", 16))
    label.pack(pady=10)

    btn1 = tk.Button(
        root,
        text="(A) Exclude Fibres Manually",
        command=lambda: [
            root.destroy(),
            ensure_segmented(),  # <- ensure df is defined
            manual_exclude(),
            show_menu()
        ],
        width=40,
        height=2
    )
    btn1.pack(pady=5)

    btn2 = tk.Button(root, text="(B) Classify Fibres (Manual or Auto)", command=lambda: [root.destroy(), classify_menu()], width=40, height=2)
    btn2.pack(pady=5)

    root.mainloop()

show_menu()