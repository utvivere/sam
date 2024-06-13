import numpy as np
import wandb

def normalize_columns(U):
    norms = np.linalg.norm(U, axis=0)
    return U / norms

# loss: 1/2m sum_{i=1}^{m} (yi - <Ai, UU^T>)^2
def compute_loss(A, y, U):
    m = len(A)
    loss = 0.0
    
    for i in range(m):
        AU = A[i] @ U @ U.T
        loss += (y[i] - np.trace(AU)) ** 2
    
    loss = (1 / (2 * m)) * loss
    return loss

# 1/m sum_{i=1}^{m} (yi - <Ai, UU^T>) * 
def compute_gradient(A, y, U):
    m = len(A)
    grad = np.zeros_like(U)
    
    for i in range(m):
        AU = A[i] @ U @ U.T
        error = y[i] - np.trace(AU)
        grad += -error * (A[i] + A[i].T) @ U
    
    grad = grad / m
    return grad

def gradient_descent(A_list, y, d, r, learning_rate=0.05, num_iterations=10, tol=1e-6):
    wandb.init(project="usam", name="gd")
    U = np.random.randn(d, r)
    U = normalize_columns(U)
    losses = []

    for iteration in range(num_iterations):
        loss = compute_loss(A_list, y, U)
        losses.append(loss)

        if loss < tol:
            print(f'Converged in {iteration} iterations.')
            break

        grad = compute_gradient(A_list, y, U)
        U -= learning_rate * grad

        # if iteration % 100 == 0:
        #    print(f'Iteration {iteration}, Loss: {loss}')

        wandb.log({"loss": loss})
    wandb.finish()

def sam_update(U, grad, learning_rate, rho, normalize):
    if normalize: # with normalization
        perturbation = rho * grad / np.linalg.norm(grad)
    else: # without normalization
        perturbation = rho * grad
    U_perturbed = U + perturbation
    return U - learning_rate * compute_gradient(A_list, y, U_perturbed), np.linalg.norm(grad)

def sam(A_list, y, d, r, rho, normalize, learning_rate=0.5, num_iterations=10, tol=1e-6):

    U = np.random.randn(d, r)
    U = normalize_columns(U)
    losses = []

    for iteration in range(num_iterations):
        loss = compute_loss(A_list, y, U)
        losses.append(loss)

        if loss < tol:
            print(f'Converged in {iteration} iterations.')
            break

        grad = compute_gradient(A_list, y, U)
        U, grad_norm = sam_update(U, grad, learning_rate, rho, normalize=normalize)

        # if iteration % 100 == 0:
        #     print(f'Iteration {iteration}, Loss: {loss}')

        wandb.log({"loss": loss, "grad_norm": grad_norm})

# Follow Yan Dai et.al. paper's setup
d = 100  
r = 5   
m = 5 * d * r 
learning_rate = 0.5 
num_iterations = 10

U_true = np.random.randn(d, r)
print('Spectral norm of U_true before normalization:', np.linalg.norm(U_true, ord=2))
U_true = normalize_columns(U_true)
X_star = U_true @ U_true.T
# check it satisfies the requirement
print('Spectral norm of U_true after normalization:', np.linalg.norm(U_true, ord=2))

A_list = [np.random.randn(d, d) for _ in range(m)]

y = np.array([np.trace(A.T @ X_star) for A in A_list]) 

# run on gradient descent
gradient_descent(A_list, y, d, r)

# vary rho
rhos = [ 0.001, 0.005, 0.01, 0.1]

for rho in rhos:
    print(f"Running USAM with rho={rho}")
    wandb.init(project="usam", name=f"usam_rho_{rho}")
    sam(A_list, y, d, r, rho=rho, normalize=False, learning_rate = learning_rate, num_iterations = num_iterations)
    wandb.finish()

    print(f"Running SAM with rho={rho}")
    wandb.init(project="usam", name=f"sam_rho_{rho}")
    sam(A_list, y, d, r, rho=rho, normalize=True, learning_rate = learning_rate, num_iterations = num_iterations)
    wandb.finish()
