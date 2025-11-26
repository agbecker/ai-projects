import numpy as np
import random
import matplotlib.pyplot as plt

def visualize_binary_matrix(matrix1, matrix2=None, figsize=(12, 6), title1="Matrix 1", title2="Matrix 2"):
    """
    Visualize one or two binary matrices as pixel images where 1 is black and 0 is white.
    
    Parameters:
    -----------
    matrix1 : numpy.ndarray or list
        First 2D array containing 0s and 1s
    matrix2 : numpy.ndarray or list, optional
        Second 2D array containing 0s and 1s. If None, only matrix1 is displayed.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (12, 6) for two matrices
    title1 : str, optional
        Title for the first matrix. Default is "Matrix 1"
    title2 : str, optional
        Title for the second matrix. Default is "Matrix 2"
    
    Returns:
    --------
    None
        Displays the visualization using matplotlib
    
    Example:
    --------
    >>> matrix1 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    >>> matrix2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]])
    >>> visualize_binary_matrix(matrix1, matrix2)
    """
    # Convert to numpy arrays
    matrix1 = np.array(matrix1)
    
    if matrix2 is None:
        # Single matrix display
        fig, ax = plt.subplots(figsize=(6, 6))
        axes = [ax]
        matrices = [matrix1]
        titles = [title1]
    else:
        # Two matrix display
        matrix2 = np.array(matrix2)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        matrices = [matrix1, matrix2]
        titles = [title1, title2]
    
    for ax, matrix, title in zip(axes, matrices, titles):
        # Ensure matrix is 2D
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim > 2:
            raise ValueError(f"Matrix must be 1D or 2D, got shape {matrix.shape}")
        
        # Display the matrix as an image
        # cmap='gray_r' uses white for 0 and black for 1 (reversed)
        ax.imshow(matrix, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(title, fontsize=14, pad=10)
        
        # Add grid for better visualization
        n, m = matrix.shape
        ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# Function to obtain a matrix from an integer
def num_to_mat(n, x):
    """
    Convert an integer to a 2D binary matrix representation.
    
    Parameters:
    -----------
    n : int
        Dimension of the output square matrix (n x n)
    x : int
        Integer to convert to binary representation
    
    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (n, n) where the i-th position contains
        the i-th bit of x in binary representation (n^2 bits total)
    
    Example:
    --------
    >>> num_to_mat(3, 11)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 1]])
    
    11 in 9-bit binary is 000001011
    """
    # Calculate total number of bits needed
    total_bits = n * n
    
    # Convert integer to binary string with n^2 bits (padded with zeros)
    binary_str = format(x, f'0{total_bits}b')
    
    # Convert binary string to list of integers
    binary_list = [int(bit) for bit in binary_str]
    
    # Reshape into (n, n) matrix
    matrix = np.array(binary_list).reshape(n, n)
    
    return matrix

# Function to output the nonogram clues for a given drawing
def get_clues(board : np.array):
    verticals = []
    horizontals = []

    for row in board:
        clue = []
        cur_count = 0
        for cell in row:
            if cell == 1:
                cur_count += 1

            elif cur_count > 0:
                clue.append(str(cur_count))
                cur_count = 0
        if cur_count > 0:
            clue.append(str(cur_count))
        if not len(clue):
            clue.append('0')
        horizontals.append(clue)

    for col in board.T:
        clue = []
        
        cur_count = 0
        for cell in col:
            if cell == 1:
                cur_count += 1

            elif cur_count > 0:
                clue.append(str(cur_count))
                cur_count = 0
        if cur_count > 0:
            clue.append(str(cur_count))
        if not len(clue):
            clue.append('0')
        verticals.append(clue)

    horizontals = [','.join(clue) for clue in horizontals]
    verticals = [','.join(clue) for clue in verticals]
    
    return '|'.join(horizontals) + '#' + '|'.join(verticals)

if __name__ == '__main__':
    
    sols = dict()
    for n in range(4,5):
        for i in range(2**(n**2)):
            mat = num_to_mat(n, i)
            clues = get_clues(mat)
            
            if clues not in sols:
                sols[clues] = [mat]
            else:
                sols[clues].append(mat)

    # Display two matrices with the same clues
    clue = random.choice([k for k in sols if len(sols[k]) > 1])
    ununique = sols[clue]
    print(clue)
    visualize_binary_matrix(ununique[0], ununique[1], 
                           title1="Solution 1", title2="Solution 2")