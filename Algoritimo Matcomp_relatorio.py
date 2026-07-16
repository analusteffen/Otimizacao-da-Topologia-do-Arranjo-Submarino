import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad

# =====================================================================
# FASE 1: Setup e Modelagem do Leito Marinho (Batimetria e Splines)
# =====================================================================

# 1.1 Gerando um grid de 10km x 10km para o campo de petróleo
x_grid = np.linspace(0, 10000, 50)
y_grid = np.linspace(0, 10000, 50)
X, Y = np.meshgrid(x_grid, y_grid)

# 1.2 Criando uma batimetria sintética Z (profundidade)
Z = 500 + 0.1 * X + 0.05 * Y - 50 * np.exp(-((X - 500)**2 + (Y - 500)**2) / 20000)

# 1.3 Criando a malha contínua usando Splines Bicúbicas
seabed_spline = RectBivariateSpline(x_grid, y_grid, Z.T)

# =====================================================================
# FASE 2: Entrada de Dados Dinâmica
# =====================================================================

def obter_dados_usuario():
    """
    Coleta as coordenadas dos poços, pesos e a posição inicial do manifold
    através do terminal.
    """
    print("\n" + "="*50)
    print(" CONFIGURAÇÃO DO ARRANJO SUBMARINO ")
    print("="*50)
    
    while True:
        try:
            num_pocos = int(input("Quantos poços deseja conectar ao manifold? "))
            if num_pocos <= 0:
                print("Por favor, insira um número maior que zero.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Digite um número inteiro.")

    wells = []
    weights = []
    
    for i in range(num_pocos):
        print(f"\n--- Dados do Poço {i+1} ---")
        while True:
            try:
                x = float(input(f"Coordenada X (0 a 10km): "))
                y = float(input(f"Coordenada Y (0 a 10km): "))
                w = float(input(f"Custo/Peso do duto (ex: 1.0): "))
                wells.append([x, y])
                weights.append(w)
                break
            except ValueError:
                print("Entrada inválida. Use apenas números (com ponto para decimais).")

    print("\n--- Posição Inicial do Manifold (Chute Inicial) ---")
    while True:
        try:
            mx = float(input("Coordenada X inicial: "))
            my = float(input("Coordenada Y inicial: "))
            manifold_inicial = [mx, my]
            break
        except ValueError:
            print("Entrada inválida. Use apenas números.")

    return np.array(wells), np.array(weights), manifold_inicial


# =====================================================================
# FASE 3: Funções Matemáticas do Modelo
# =====================================================================

def calculate_pipe_length(well_pos, manifold_pos):
    """
    Calcula o comprimento real de um duto repousando no leito marinho 
    entre um poço e o manifold usando integração numérica.
    """
    xp, yp = well_pos
    xm, ym = manifold_pos
    
    dx_dt = xm - xp
    dy_dt = ym - yp
    
    def integrand(t):
        xt = xp + t * dx_dt
        yt = yp + t * dy_dt
        
        ds_dx = seabed_spline(xt, yt, dx=1, dy=0, grid=False)
        ds_dy = seabed_spline(xt, yt, dx=0, dy=1, grid=False)
        
        dz_dt = ds_dx * dx_dt + ds_dy * dy_dt
        
        return np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt.item()**2)
    
    length, _ = quad(integrand, 0, 1)
    return length


def cost_function(manifold_pos, wells, weights):
    """
    Calcula o custo total do layout somando (comprimento do duto * peso)
    para todos os poços conectados.
    """
    total_cost = 0
    for i, well in enumerate(wells):
        length = calculate_pipe_length(well, manifold_pos)
        total_cost += weights[i] * length
    return total_cost


# =====================================================================
# FASE 4: Otimização por Descida de Gradiente
# =====================================================================

def gradient_descent(start_pos, wells, weights, learning_rate=0.5, tol=1e-4, max_iter=150):
    """
    Minimiza a função custo alterando a posição do manifold iterativamente.
    """
    pos = np.array(start_pos, dtype=float)
    h = 1e-4  
    
    print("\n" + "="*50)
    print(f"Iniciando otimização. Posição inicial: ({pos[0]:.2f}, {pos[1]:.2f})")
    print("="*50)
    
    for iteration in range(max_iter):
        current_cost = cost_function(pos, wells, weights)
        
        # Derivadas parciais numéricas
        cost_x_plus = cost_function([pos[0] + h, pos[1]], wells, weights)
        cost_x_minus = cost_function([pos[0] - h, pos[1]], wells, weights)
        grad_x = (cost_x_plus - cost_x_minus) / (2 * h)
        
        cost_y_plus = cost_function([pos[0], pos[1] + h], wells, weights)
        cost_y_minus = cost_function([pos[0], pos[1] - h], wells, weights)
        grad_y = (cost_y_plus - cost_y_minus) / (2 * h)
        
        grad = np.array([grad_x, grad_y])
        
        # Passo de Descida
        new_pos = pos - learning_rate * grad
        new_cost = cost_function(new_pos, wells, weights)
        
        print(f"Iter {iteration+1:03d} | Pos: ({new_pos[0]:.2f}, {new_pos[1]:.2f}) | Custo: {new_cost:.2f}")
        
        # Critério de Parada
        if abs(current_cost - new_cost) < tol:
            print("\n>>> Convergência atingida! Mínimo local encontrado.")
            return new_pos, new_cost
            
        pos = new_pos
        
    print("\n>>> Limite máximo de iterações atingido sem convergência completa.")
    return pos, cost_function(pos, wells, weights)


# =====================================================================
# EXECUÇÃO DO PROJETO
# =====================================================================

if __name__ == "__main__":
    # 1. Obter dados dinamicamente do usuário
    wells_input, weights_input, initial_guess_input = obter_dados_usuario()
    
    # 2. Executar o otimizador passando os dados coletados
    optimal_pos, min_cost = gradient_descent(
        start_pos=initial_guess_input, 
        wells=wells_input, 
        weights=weights_input, 
        learning_rate=0.8
    )
    
    # 3. Exibir Resultados
    print("-" * 50)
    print("RESULTADO FINAL DA OTIMIZAÇÃO")
    print("-" * 50)
    print(f"Coordenada Ideal do Manifold: X = {optimal_pos[0]:.2f} m, Y = {optimal_pos[1]:.2f} m")
    print(f"Custo Total Ponderado:        {min_cost:.2f} unidades")