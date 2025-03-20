import numpy as np
import gudhi
import os
import matplotlib.pyplot as plt
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from gudhi.wasserstein import wasserstein_distance
from ripser import ripser
from collections import defaultdict
from gtda.diagrams import PersistenceLandscape
from scipy import linalg
from persim import plot_diagrams
import warnings
warnings.filterwarnings("ignore")


def calcultae_bary_center(dgms):

    dgms = [np.array(dgm) for dgm in dgms]

    b, log = bary(dgms, 
         init=0,
         verbose=True)  
    
    return b

def calculate_wasserstein_distance(dgm1, dgm2):

    dgm1 = np.array(dgm1)
    dgm2 = np.array(dgm2)

    d = wasserstein_distance(dgm1, dgm2, matching=True, order=2)[0]
    
    return d

def extract_center_coordinates(file_path):

    point_cloud = np.load(file_path)
    #print(point_cloud.shape)

    def get_center_coordinates_single_channel(channel):
        # Label connected components (cells)
        from scipy.ndimage import label, center_of_mass
        labeled_array, num_features = label(channel)
        
        if num_features == 0:
            return np.empty((0, 2))
            
        # Calculate center of mass for each labeled region
        centers = np.array([center_of_mass(channel, labeled_array, index)
                          for index in range(1, num_features + 1)])
        
        return centers

    # Ensure the input is a numpy array
    point_cloud = np.array(point_cloud)
    
    # Check input dimensions
    if len(point_cloud.shape) != 3 or point_cloud.shape[2] != 3:
        raise ValueError("Input must be a 3-channel point cloud with shape (height, width, 3)")
    
    # Process each channel
    coordinates = []
    for channel in range(3):
        channel_coords = get_center_coordinates_single_channel(point_cloud[..., channel])
        coordinates.append(channel_coords)
    
    return coordinates

def create_layout_mapping(ref_folder, gen_folder):
    # Dictionary to store the mapping
    layout_mapping = defaultdict(list)
    
    # Get list of reference layouts
    ref_layouts = os.listdir(ref_folder)
    
    # Get list of generated layouts
    gen_layouts = os.listdir(gen_folder)
    
    # Create mapping
    for ref_layout in ref_layouts:
        if ref_layout.endswith('.npy'):
            ref_base = ref_layout.rsplit('.', 1)[0]
        #ref_base = ref_layout.rsplit('.', 1)[0]  # Remove file extension
        for gen_layout in gen_layouts:
            if gen_layout.startswith(ref_base):
                layout_mapping[ref_layout].append(gen_layout)
    
    return layout_mapping

def extract_1_dim_pd(point_cloud):
    alpha_complex = gudhi.RipsComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree(max_dimension=2)

    # Compute persistent homology
    persistence = simplex_tree.persistence()
    
    # Separate 0-dim and 1-dim persistent homology
    ph_0dim = [p[1] for p in persistence if p[0] == 0]
    ph_1dim = [p[1] for p in persistence if p[0] == 1]
    
    return ph_1dim

def compute_1d_persistent_homology(point_cloud):
    """
    Compute 1-dimensional persistent homology for a given point cloud using an alpha complex.
    
    :param point_cloud: numpy array of shape (n_points, dimension)
    :return: list of 1-dimensional persistence pairs
    """
    # Create an alpha complex from the point cloud
    alpha_complex = gudhi.AlphaComplex(points=point_cloud, precision = 'exact')
    
    # Compute the simplex tree
    simplex_tree = alpha_complex.create_simplex_tree()
    
    # Compute persistent homology
    persistence = simplex_tree.persistence()
    
    # Extract only the 1-dimensional persistent homology
    ph_1dim = [p for p in persistence if p[0] == 1]
    
    return ph_1dim

def convert_tuple_to_list(pds):
    """
    Convert a list of tuples (persistence diagram) to a list of lists.
    
    :param pds: A list of tuples representing the persistence diagram
    :return: A list of lists representing the persistence diagram
    """
    result = []
    for pd in pds:
        # Convert each tuple to a list
        pd_list = list(pd)
        # Replace 'inf' with float('inf') if present
        pd_list = [float('inf') if x == 'inf' else x for x in pd_list]
        result.append(pd_list)
    return result

def vectorize_pl(diagram):
    pl = PersistenceLandscape()

    #print(diagram.shape)

    if diagram.shape[0] == 0:
        return np.zeros(100)
    
    dgms = np.column_stack((
        diagram,
        np.ones((diagram.shape[0], 1))
    ))
    dgms = [dgms]

    pl_vector = pl.fit_transform(dgms).flatten()
    
    return pl_vector

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Frechet distance between two multivariate Gaussians.
    
    Parameters:
    mu1, mu2: Mean vectors
    sigma1, sigma2: Covariance matrices
    eps: Small epsilon to avoid singular matrix

    Returns:
    Frechet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def visualize_and_save_pd(pd, file_path, title):
    plt.figure(figsize=(8, 8))
    plot_diagrams([pd], show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def calculate_covariance_matrix(vectors):
    """
    Calculate the covariance matrix of a set of vectors.
    """
    return np.cov(vectors, rowvar=False)

def main():

    print("Starting TopoFD evaluation...")
    ref_folder = "path/to/reference/folder"
    gen_root_path = "path/to/generated/folder"
    gen_folder = os.path.join(gen_root_path, "npy")

    layout_mapping = create_layout_mapping(ref_folder, gen_folder)

    
    ref_file_list = os.listdir(ref_folder)
    ref_file_list = [f for f in ref_file_list if f.endswith('.npy')]

    pd1_0_ref, pd1_1_ref, pd1_2_ref = [], [], []
    pd1_0_gen, pd1_1_gen, pd1_2_gen = [], [], []

    for ref_file_name in ref_file_list:
        gen_file_name = layout_mapping[ref_file_name][0]

        ref_file_path = os.path.join(ref_folder, ref_file_name)
        gen_file_path = os.path.join(gen_folder, gen_file_name)

        ref_coordinates = extract_center_coordinates(ref_file_path)
        gen_coordinates = extract_center_coordinates(gen_file_path)

        pd1_0_ref.append(extract_1_dim_pd(ref_coordinates[0]))
        pd1_1_ref.append(extract_1_dim_pd(ref_coordinates[1]))
        pd1_2_ref.append(extract_1_dim_pd(ref_coordinates[2]))

        pd1_0_gen.append(extract_1_dim_pd(gen_coordinates[0]))
        pd1_1_gen.append(extract_1_dim_pd(gen_coordinates[1]))
        pd1_2_gen.append(extract_1_dim_pd(gen_coordinates[2]))

    bary_0_ref = calcultae_bary_center(pd1_0_ref)
    bary_1_ref = calcultae_bary_center(pd1_1_ref)
    bary_2_ref = calcultae_bary_center(pd1_2_ref)

    bary_0_gen = calcultae_bary_center(pd1_0_gen)
    bary_1_gen = calcultae_bary_center(pd1_1_gen)
    bary_2_gen = calcultae_bary_center(pd1_2_gen)

    vectorized_bary_0_ref = vectorize_pl(bary_0_ref)
    vectorized_bary_1_ref = vectorize_pl(bary_1_ref)
    vectorized_bary_2_ref = vectorize_pl(bary_2_ref)

    vectorized_bary_0_gen = vectorize_pl(bary_0_gen)
    vectorized_bary_1_gen = vectorize_pl(bary_1_gen)
    vectorized_bary_2_gen = vectorize_pl(bary_2_gen)

    vectorized_pd1_0_ref = [vectorize_pl(np.array(pd)) for pd in pd1_0_ref]
    vectorized_pd1_1_ref = [vectorize_pl(np.array(pd)) for pd in pd1_1_ref]
    vectorized_pd1_2_ref = [vectorize_pl(np.array(pd)) for pd in pd1_2_ref]

    vectorized_pd1_0_gen = [vectorize_pl(np.array(pd)) for pd in pd1_0_gen]
    vectorized_pd1_1_gen = [vectorize_pl(np.array(pd)) for pd in pd1_1_gen]
    vectorized_pd1_2_gen = [vectorize_pl(np.array(pd)) for pd in pd1_2_gen]

    # Calculate the covariance matrix
    cov_matrix_0_ref = calculate_covariance_matrix(vectorized_pd1_0_ref)
    cov_matrix_1_ref = calculate_covariance_matrix(vectorized_pd1_1_ref)
    cov_matrix_2_ref = calculate_covariance_matrix(vectorized_pd1_2_ref)

    assert cov_matrix_0_ref.shape == (100, 100), "Covariance matrix should be 100x100"

    cov_matrix_0_gen = calculate_covariance_matrix(vectorized_pd1_0_gen)
    cov_matrix_1_gen = calculate_covariance_matrix(vectorized_pd1_1_gen)
    cov_matrix_2_gen = calculate_covariance_matrix(vectorized_pd1_2_gen)

    assert cov_matrix_0_gen.shape == (100, 100), "Covariance matrix should be 100x100"

    # Calculate the Frechet distance
    fd_0 = calculate_frechet_distance(vectorized_bary_0_ref, cov_matrix_0_ref, vectorized_bary_0_gen, cov_matrix_0_gen)
    fd_1 = calculate_frechet_distance(vectorized_bary_1_ref, cov_matrix_1_ref, vectorized_bary_1_gen, cov_matrix_1_gen)
    fd_2 = calculate_frechet_distance(vectorized_bary_2_ref, cov_matrix_2_ref, vectorized_bary_2_gen, cov_matrix_2_gen)

    print(f"TopoFD for channel 0: {fd_0}")
    print(f"TopoFD for channel 1: {fd_1}")
    print(f"TopoFD for channel 2: {fd_2}")

    fid_total = (fd_0 + fd_1 + fd_2) / 3

    print(f"Average TopoFD: {fid_total}")

    # Save results to a file
    result_file = os.path.join(os.path.dirname(gen_folder), "topo_fd_results.txt")
    with open(result_file, "w") as f:
        f.write(f"FD for channel 0: {fd_0}\n")
        f.write(f"FD for channel 1: {fd_1}\n")
        f.write(f"FD for channel 2: {fd_2}\n")
        f.write(f"Total FD: {fid_total}\n")

    print("TopoFD evaluation completed.")

if __name__ == "__main__":
    main()


