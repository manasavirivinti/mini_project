import numpy as np

def fitness_function(weights, data, labels):
    print("Shape of data:", data.shape)
    print("Shape of weights before reshape:", weights.shape)
    
    # Reshape weights to match input features
    weights = weights.reshape((data.shape[1], -1))
    
    print("Shape of weights after reshape:", weights.shape)
    
    # Perform prediction using linear combination followed by sigmoid
    predictions = 1 / (1 + np.exp(-(data @ weights)))
    
    # Reshape predictions to match labels shape
    predictions = predictions.reshape(-1, 64)  # Assuming 64 classes or outputs
    
    # Convert predictions to binary classification (threshold = 0.5)
    predictions = (predictions > 0.5).astype(int)
    
    # Convert predictions to class labels (e.g., using argmax for multi-class)
    predictions = np.argmax(predictions, axis=1)
    
    return np.mean(predictions == labels)


# Initialize population
def initialize_population(pop_size, weight_shape):
    return np.random.rand(pop_size, np.prod(weight_shape))

# Update rules for GJO algorithm
def update_population(population, alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness):
    new_population = np.copy(population)
    for i in range(len(population)):
        r1, r2 = np.random.rand(), np.random.rand()
        A1, A2 = 2 * r1 * alpha - alpha, 2 * r2 * beta - beta
        
        C1, C2 = 2 * r1, 2 * r2
        D_alpha = np.abs(C1 * alpha - population[i])
        D_beta = np.abs(C2 * beta - population[i])
        
        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        
        new_population[i] = (X1 + X2) / 2
    
    return new_population

def run_gjo(X_train, y_train, weight_shape):
    # Hyperparameters
    pop_size = 50
    iterations = 100

    # Initialize population
    population = initialize_population(pop_size, weight_shape)

    # Initialize alpha, beta, delta and their fitness values
    alpha = population[0]
    beta = population[1]
    delta = population[2]
    
    alpha_fitness = fitness_function(alpha, X_train, y_train)
    beta_fitness = fitness_function(beta, X_train, y_train)
    delta_fitness = fitness_function(delta, X_train, y_train)

    # Optimization loop
    for iteration in range(iterations):
        for i in range(pop_size):
            fitness = fitness_function(population[i], X_train, y_train)
            if fitness > alpha_fitness:
                alpha, alpha_fitness = population[i], fitness
            elif fitness > beta_fitness:
                beta, beta_fitness = population[i], fitness
            elif fitness > delta_fitness:
                delta, delta_fitness = population[i], fitness
        
        # Update population based on alpha, beta, delta
        population = update_population(population, alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness)

    best_solution = alpha  # Best solution found
    return best_solution
