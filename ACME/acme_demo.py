import os
os.environ["OMP_NUM_THREADS"] = "1"   # <-- PRIMERO DE TODO

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imageio.v2 as imageio
import random
from copy import deepcopy
from PIL import Image

# --- Configuration ---
POP_SIZE = 50
NGEN = 15
K_CLUSTERS = 5
LS_ITERS = 50
BOUNDS = [-10, 10]
memetic = 1
GIF_NAME = "acme_demonstration4.gif"
FRAMES_DIR = "frames"

# --- Griewank Function ---
def griewank(x):
    # x is an array of shape (N, 2) or (2,)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    sum_sq = np.sum(x**2, axis=1) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
    return sum_sq - prod_cos + 1

def griewank_grad(x):
    """
    Gradiente exacto de la función de Griewank para un vector x de dimensión 2 (o n).
    """
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    idx = np.arange(1, n + 1)

    # Derivada de la parte cuadrática: sum(x_i^2)/4000 -> x_i / 2000
    grad_S = x / 2000.0

    # Parte del producto de cosenos
    scaled = x / np.sqrt(idx)
    cos_vals = np.cos(scaled)
    sin_vals = np.sin(scaled)
    P = np.prod(cos_vals)

    grad_P = np.zeros_like(x)
    for k in range(n):
        # producto de todos los cosenos excepto el k-ésimo
        if np.isclose(cos_vals[k], 0.0):
            prod_others = np.prod(np.delete(cos_vals, k))
        else:
            prod_others = P / cos_vals[k]
        grad_P[k] = prod_others * (-sin_vals[k] / np.sqrt(idx[k]))

    # f(x) = S - P + 1  => ∇f = ∇S - ∇P
    grad_f = grad_S - grad_P
    return grad_f


def plot_frame(pop, title, filename,
               clusters=None, reps=None, ls_paths=None,
               global_best=None):
    plt.figure(figsize=(10, 8))
    
    # Background Contour
    x = np.linspace(BOUNDS[0], BOUNDS[1], 100)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = griewank(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Fitness (Griewank)')
    
    # --- Mejor de la población actual ---
    fits = griewank(pop)
    best_idx = np.argmin(fits)
    best_point = pop[best_idx]
    best_fit = fits[best_idx]
    
    # --- Mejor global (si existe) ---
    glob_point, glob_fit = None, None
    if global_best is not None:
        glob_point, glob_fit = global_best

    # Population
    if clusters is None:
        plt.scatter(pop[:, 0], pop[:, 1], c='white',
                    edgecolor='black', s=50, label='Population')
    else:
        plt.scatter(pop[:, 0], pop[:, 1], c=clusters, cmap='Set1',
                    edgecolor='black', s=50, label='Clustered Pop')
        
    # Representatives
    if reps is not None:
        reps = np.array(reps)
        plt.scatter(reps[:, 0], reps[:, 1], c='red', marker='*', s=200,
                    edgecolor='white', label='Representatives', zorder=10)
        
    # Local Search Paths
    if ls_paths is not None:
        for path in ls_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.8)
            plt.scatter(path[-1, 0], path[-1, 1], c='yellow', marker='*',
                        s=250, edgecolor='black', zorder=11)

    # Marcar mejor de la población
    plt.scatter(best_point[0], best_point[1],
                marker='X', s=250, c='lime', edgecolor='black',
                zorder=12, label='Best in pop')

    # Marcar mejor global
    if glob_point is not None:
        plt.scatter(glob_point[0], glob_point[1],
                    marker='X', s=250, c='magenta', edgecolor='black',
                    zorder=13, label='Best global')

    plt.title(title)
    plt.xlim(BOUNDS)
    plt.ylim(BOUNDS)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Dejar más hueco arriba para el texto
    # (baja un poco el eje dentro de la figura)
    plt.subplots_adjust(top=0.78)

    # Texto arriba con ambos fitness
    ax = plt.gca()
    text_lines = []
    if glob_point is not None:
        text_lines.append(
            f"Best global: f(x) = {glob_fit:.6f} at ({glob_point[0]:.3f}, {glob_point[1]:.3f})"
        )
    text_lines.append(
        f"Best pop: f(x) = {best_fit:.6f} at ({best_point[0]:.3f}, {best_point[1]:.3f})"
    )

    ax.text(
        0.5, 1.10,                # <-- más arriba (antes 1.03)
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # recuadro para que se lea mejor
    )
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# --- ACME Logic ---

def run_demo():
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
        
    # 1. Initialize Population
    pop = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, 2))
    frame_count = 0
    
    filenames = []

    # Mejor global (punto y fitness)
    global_best_point = None
    global_best_fit = np.inf

    def update_global_best_from_pop(pop, global_best_point, global_best_fit):
        fits = griewank(pop)
        idx = np.argmin(fits)
        best_point = pop[idx]
        best_fit = fits[idx]
        if best_fit < global_best_fit:
            global_best_fit = best_fit
            global_best_point = best_point.copy()
        return global_best_point, global_best_fit

    for gen in range(NGEN):
        print(f"Generation {gen}")
        
        # --- Step 1: Standard Population ---
        global_best_point, global_best_fit = update_global_best_from_pop(
            pop, global_best_point, global_best_fit
        )
        fname = f"{FRAMES_DIR}/gen_{gen:03d}_01_pop.png"
        plot_frame(
            pop,
            f"Gen {gen}: Population",
            fname,
            global_best=(global_best_point, global_best_fit)
        )
        filenames.append(fname)

        labels = None  # por defecto, sin clustering esta gen

        if gen % memetic == 0:
            # --- Step 2: Clustering ---
            kmeans = KMeans(n_clusters=K_CLUSTERS, n_init=10)
            labels = kmeans.fit_predict(pop)
            centroids = kmeans.cluster_centers_
            
            # Find representatives (closest to centroid)
            reps = []
            for k in range(K_CLUSTERS):
                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_points = pop[cluster_indices]
                dists = np.linalg.norm(cluster_points - centroids[k], axis=1)
                best_in_cluster_idx = cluster_indices[np.argmin(dists)]
                reps.append(pop[best_in_cluster_idx].copy())
                
            fname = f"{FRAMES_DIR}/gen_{gen:03d}_02_clusters.png"
            global_best_point, global_best_fit = update_global_best_from_pop(
                pop, global_best_point, global_best_fit
            )
            plot_frame(
                pop,
                f"Gen {gen}: K-Means Clustering",
                fname,
                clusters=labels,
                global_best=(global_best_point, global_best_fit)
            )
            filenames.append(fname)
            
            fname = f"{FRAMES_DIR}/gen_{gen:03d}_03_reps.png"
            global_best_point, global_best_fit = update_global_best_from_pop(
                pop, global_best_point, global_best_fit
            )
            plot_frame(
                pop,
                f"Gen {gen}: Representatives Selection",
                fname,
                clusters=labels,
                reps=reps,
                global_best=(global_best_point, global_best_fit)
            )
            filenames.append(fname)
            
            # --- Step 3: Local Search on Representatives ---
            # IMPORTANTE: NO se modifica la población, solo se usa para mejorar global_best
            ls_paths = []
            
            for rep in reps:
                path = [rep.copy()]
                curr = rep.copy()
                curr_fit = griewank(curr)[0]

                # Parámetros de descenso por gradiente
                lr = 1.0 * (0.9 ** gen)   # learning rate que decae con la generación
                tol = 1e-6

                for _ in range(LS_ITERS):
                    grad = griewank_grad(curr)
                    grad_norm = np.linalg.norm(grad)

                    if grad_norm < tol:
                        break

                    # Paso de descenso por gradiente
                    curr = curr - lr * grad
                    curr = np.clip(curr, BOUNDS[0], BOUNDS[1])

                    new_fit = griewank(curr)[0]

                    if new_fit <= curr_fit:
                        curr_fit = new_fit
                        path.append(curr.copy())

                ls_paths.append(path)

                # === Aquí solo registramos la solución si mejora el mejor global ===
                if curr_fit < global_best_fit:
                    global_best_fit = curr_fit
                    global_best_point = curr.copy()

            fname = f"{FRAMES_DIR}/gen_{gen:03d}_04_ls.png"
            plot_frame(
                pop,  # OJO: seguimos pintando la población original
                f"Gen {gen}: Local Search on Representatives",
                fname,
                clusters=labels,
                reps=reps,
                ls_paths=ls_paths,
                global_best=(global_best_point, global_best_fit)
            )
            filenames.append(fname)
        
        # --- Step 4: NO actualizar la población con el LS ---
        # new_pop es simplemente una copia de la población actual
        new_pop = pop.copy()
        new_pop = np.clip(new_pop, BOUNDS[0], BOUNDS[1])

        fname = f"{FRAMES_DIR}/gen_{gen:03d}_05_update.png"
        global_best_point, global_best_fit = update_global_best_from_pop(
            new_pop, global_best_point, global_best_fit
        )
        plot_frame(
            new_pop,
            f"Gen {gen}: Population (before GA)",
            fname,
            clusters=labels,
            global_best=(global_best_point, global_best_fit)
        )
        filenames.append(fname)

        pop = new_pop
        
        # --- Step 5: Standard GA Operations (Selection, Crossover, Mutation) ---
        next_pop = []
        fitnesses = griewank(pop)
        
        for _ in range(POP_SIZE):
            candidates_idx = np.random.choice(POP_SIZE, 3, replace=False)
            best_idx = candidates_idx[np.argmin(fitnesses[candidates_idx])]
            next_pop.append(pop[best_idx].copy())
            
        next_pop = np.array(next_pop)
        
        # Crossover (Arithmetic)
        for i in range(0, POP_SIZE, 2):
            if i+1 < POP_SIZE and np.random.rand() < 0.8:
                alpha = np.random.rand()
                c1 = alpha * next_pop[i] + (1-alpha) * next_pop[i+1]
                c2 = (1-alpha) * next_pop[i] + alpha * next_pop[i+1]
                next_pop[i] = c1
                next_pop[i+1] = c2
                
        # Mutation (Gaussian)
        mutation_strength = 2.0
        mask = np.random.rand(POP_SIZE) < 0.2
        noise = np.random.normal(0, mutation_strength, (np.sum(mask), 2))
        next_pop[mask] += noise
        next_pop = np.clip(next_pop, BOUNDS[0], BOUNDS[1])
        
        pop = next_pop

    # Create GIF
    print("Creating GIF...")

    frames = [Image.open(f) for f in filenames]

    frames[0].save(
        GIF_NAME,
        save_all=True,
        append_images=frames[1:],
        duration=1500,   # ⬅ 1500 ms = 1.5 segundos por frame
        loop=0
    )

    print(f"Done! Saved to {GIF_NAME}")
    for filename in filenames:
        os.remove(filename)
    os.rmdir(FRAMES_DIR)
    print(f"Done! Saved to {GIF_NAME}")

if __name__ == "__main__":
    run_demo()
