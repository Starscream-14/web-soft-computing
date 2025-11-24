from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import random
import string

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fuzzy')
def fuzzy_page():
    return render_template('html/fuzzy.html')

@app.route('/ann')
def ann_page():
    return render_template('html/ann.html')

@app.route('/genetic')
def genetic_page():
    return render_template('html/genetic.html')

@app.route('/genetic2')
def genetic2_page():
    return render_template('html/genetic2.html')


@app.route('/tsp')
def tsp_page():
    return render_template('html/tsp.html')

def fuzzy_membership_temp(temp):
    dingin = max(0, min(1, (20 - temp) / 10)) if temp <= 20 else 0
    normal = max(0, min((temp - 15) / 10, (35 - temp) / 10))
    panas = max(0, min((temp - 30) / 10, 1)) if temp >= 30 else 0
    return {"dingin": round(dingin, 2), "normal": round(normal, 2), "panas": round(panas, 2)}


def fuzzy_membership_humidity(hum):
    kering = max(0, min(1, (40 - hum) / 20)) if hum <= 40 else 0
    sedang = max(0, min((hum - 30) / 20, (70 - hum) / 20))
    lembab = max(0, min((hum - 60) / 20, 1)) if hum >= 60 else 0
    return {"kering": round(kering, 2), "sedang": round(sedang, 2), "lembab": round(lembab, 2)}


def fuzzy_inference(temp_vals, hum_vals):
    comfort_high = min(temp_vals["normal"], hum_vals["sedang"])
    comfort_medium = max(
        min(temp_vals["normal"], hum_vals["kering"]),
        min(temp_vals["normal"], hum_vals["lembab"])
    )
    comfort_low = max(temp_vals["dingin"], temp_vals["panas"])
    
    if comfort_high + comfort_medium + comfort_low == 0:
        comfort_score = 50
    else:
        comfort_score = (comfort_high * 80 + comfort_medium * 50 + comfort_low * 20) / (
            comfort_high + comfort_medium + comfort_low
        )
    
    return round(comfort_score, 1)


@app.route('/api/fuzzy', methods=['POST'])
def fuzzy_demo():
    try:
        data = request.get_json()
        temp = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 60))
        
        temp_membership = fuzzy_membership_temp(temp)
        hum_membership = fuzzy_membership_humidity(humidity)
        comfort = fuzzy_inference(temp_membership, hum_membership)
        
        if comfort >= 70:
            label = "Sangat Nyaman"
        elif comfort >= 50:
            label = "Cukup Nyaman"
        else:
            label = "Kurang Nyaman"
        
        result = {
            "temperature": temp,
            "humidity": humidity,
            "temp_membership": temp_membership,
            "humidity_membership": hum_membership,
            "comfort_score": comfort,
            "comfort_label": label,
            "explanation": f"Dengan suhu {temp}°C dan kelembaban {humidity}%, "
                          f"sistem fuzzy menghitung tingkat kenyamanan sebesar {comfort}/100 ({label}). "
                          f"Derajat keanggotaan suhu: {temp_membership}, kelembaban: {hum_membership}."
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


class SimpleANN:
    def __init__(self):
        self.w_input_hidden = np.array([[0.5, -0.3, 0.8], [0.2, 0.6, -0.4]])
        self.b_hidden = np.array([0.1, -0.2, 0.3])
        self.w_hidden_output = np.array([[0.7], [-0.5], [0.9]])
        self.b_output = np.array([0.2])
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs):
        hidden_input = np.dot(inputs, self.w_input_hidden) + self.b_hidden
        hidden_output = self.sigmoid(hidden_input)
        
        final_input = np.dot(hidden_output, self.w_hidden_output) + self.b_output
        final_output = self.sigmoid(final_input)
        
        return final_output[0], hidden_output


ann_model = SimpleANN()


@app.route('/api/ann', methods=['POST'])
def ann_demo():
    try:
        data = request.get_json()
        x1 = float(data.get('x1', 0.5))
        x2 = float(data.get('x2', 0.3))
        
        inputs = np.array([x1, x2])
        output, hidden = ann_model.forward(inputs)
        
        result = {
            "inputs": [x1, x2],
            "hidden_layer": hidden.tolist(),
            "output": round(float(output), 4),
            "explanation": f"Input [{x1}, {x2}] diproses melalui jaringan saraf dengan 3 neuron tersembunyi. "
                          f"Aktivasi hidden layer: {[round(h, 3) for h in hidden]}. "
                          f"Output akhir (setelah sigmoid): {round(float(output), 4)}. "
                          f"Ini adalah prediksi dari model neural network sederhana yang sudah dilatih."
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


class GeneticAlgorithm:
    def __init__(self, target, population_size=100, mutation_rate=0.01):
        self.target = target.upper()
        self.target_len = len(target)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genes = string.ascii_uppercase + " "
    
    def create_individual(self):
        return ''.join(random.choices(self.genes, k=self.target_len))
    
    def fitness(self, individual):
        return sum(1 for a, b in zip(individual, self.target) if a == b)
    
    def crossover(self, parent1, parent2):
        point = random.randint(1, self.target_len - 1)
        return parent1[:point] + parent2[point:]
    
    def mutate(self, individual):
        individual_list = list(individual)
        for i in range(len(individual_list)):
            if random.random() < self.mutation_rate:
                individual_list[i] = random.choice(self.genes)
        return ''.join(individual_list)
    
    def run(self, max_generations=1000):
        population = [self.create_individual() for _ in range(self.population_size)]
        history = []
        
        for generation in range(max_generations):
            fitness_scores = [(ind, self.fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_individual, best_fitness = fitness_scores[0]
            history.append({
                "generation": generation + 1,
                "best": best_individual,
                "fitness": best_fitness
            })
            
            if best_fitness == self.target_len:
                return {
                    "success": True,
                    "generations": generation + 1,
                    "best_individual": best_individual,
                    "history": history[-10:] if len(history) > 10 else history
                }
            
            parents = [ind for ind, _ in fitness_scores[:self.population_size // 2]]
            
            new_population = parents.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        best_individual, best_fitness = max(
            [(ind, self.fitness(ind)) for ind in population],
            key=lambda x: x[1]
        )
        
        return {
            "success": False,
            "generations": max_generations,
            "best_individual": best_individual,
            "fitness": best_fitness,
            "history": history[-10:]
        }


@app.route('/api/genetic', methods=['POST'])
def genetic_demo():
    try:
        data = request.get_json()
        target = data.get('target', 'HELLO').upper()
        max_gen = int(data.get('generations', 100))
        
        if len(target) > 20:
            return jsonify({"error": "Target terlalu panjang (max 20 karakter)"}), 400
        
        ga = GeneticAlgorithm(target, population_size=100, mutation_rate=0.01)
        result = ga.run(max_generations=max_gen)
        
        if result["success"]:
            explanation = (
                f"Algoritma genetika berhasil menemukan string '{target}' dalam "
                f"{result['generations']} generasi. Proses melibatkan seleksi individu terbaik, "
                f"crossover (perkawinan), dan mutasi acak untuk evolusi populasi."
            )
        else:
            explanation = (
                f"Setelah {result['generations']} generasi, algoritma genetika menemukan "
                f"'{result['best_individual']}' dengan fitness {result['fitness']}/{len(target)}. "
                f"Tingkatkan jumlah generasi untuk hasil lebih baik."
            )
        
        result["target"] = target
        result["explanation"] = explanation
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- Genetic Algorithm 2: Knapsack problem variant ---
items_knapsack = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2},
}
capacity_default = 15


def decode_knapsack(chromosome, items_map):
    item_list = list(items_map.keys())
    total_weight = 0
    total_value = 0
    chosen = []
    for gene, name in zip(chromosome, item_list):
        if gene == 1:
            total_weight += items_map[name]['weight']
            total_value += items_map[name]['value']
            chosen.append(name)
    return chosen, total_weight, total_value


def fitness_knapsack(chromosome, items_map, capacity):
    _, w, v = decode_knapsack(chromosome, items_map)
    return v if w <= capacity else 0


def roulette_selection(population, fitnesses):
    total = sum(fitnesses)
    if total == 0:
        return random.choice(population)[:]
    pick = random.uniform(0, total)
    current = 0.0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom[:]
    return population[-1][:]


def crossover_single(p1, p2):
    if len(p1) < 2:
        return p1[:], p2[:]
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def mutate_flip(chrom, mutation_rate):
    return [1 - g if random.random() < mutation_rate else g for g in chrom]


@app.route('/api/genetic2', methods=['POST'])
def genetic2_demo():
    try:
        data = request.get_json() or {}
        pop_size = int(data.get('pop_size', 10))
        generations = int(data.get('generations', 10))
        crossover_rate = float(data.get('crossover_rate', 0.8))
        mutation_rate = float(data.get('mutation_rate', 0.1))
        elitism = bool(data.get('elitism', True))
        capacity = int(data.get('capacity', capacity_default))

        n_items = len(items_knapsack)
        population = [[random.randint(0, 1) for _ in range(n_items)] for _ in range(pop_size)]

        history = []
        for gen in range(generations):
            fitnesses = [fitness_knapsack(ch, items_knapsack, capacity) for ch in population]
            best_idx = fitnesses.index(max(fitnesses))
            best_ch = population[best_idx][:]
            chosen, w, v = decode_knapsack(best_ch, items_knapsack)
            history.append({'generation': gen + 1, 'best': best_ch, 'weight': w, 'value': v})

            new_pop = []
            if elitism:
                new_pop.append(best_ch[:])

            while len(new_pop) < pop_size:
                p1 = roulette_selection(population, fitnesses)
                p2 = roulette_selection(population, fitnesses)
                if random.random() < crossover_rate:
                    c1, c2 = crossover_single(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1 = mutate_flip(c1, mutation_rate)
                c2 = mutate_flip(c2, mutation_rate)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

            population = new_pop[:pop_size]

        fitnesses = [fitness_knapsack(ch, items_knapsack, capacity) for ch in population]
        best_idx = fitnesses.index(max(fitnesses))
        best_ch = population[best_idx][:]
        chosen, w, v = decode_knapsack(best_ch, items_knapsack)

        result = {
            'success': True,
            'best_chromosome': best_ch,
            'chosen_items': chosen,
            'total_weight': w,
            'total_value': v,
            'capacity': capacity,
            'history': history[-10:]
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# -----------------------
# TSP (Traveling Salesman) API
# -----------------------
# Default small instance (5 cities) used for demo
cities_tsp = ['A', 'B', 'C', 'D', 'E']
dist_matrix_tsp = np.array([
    [0, 7, 5, 9, 9],
    [7, 0, 7, 2, 8],
    [5, 7, 0, 4, 3],
    [9, 2, 4, 0, 6],
    [9, 8, 3, 6, 0]
], dtype=float)


def route_distance_tsp(route):
    return sum(dist_matrix_tsp[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))


def create_individual_tsp(n):
    ind = list(range(n))
    random.shuffle(ind)
    return ind


def initial_population_tsp(size, n):
    return [create_individual_tsp(n) for _ in range(size)]


def tournament_selection_tsp(pop, k):
    ksel = random.sample(pop, min(k, len(pop)))
    return min(ksel, key=lambda ind: route_distance_tsp(ind))


def ordered_crossover_tsp(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [-1] * len(p1)
    child[a:b+1] = p1[a:b+1]
    p2_idx = 0
    for i in range(len(p1)):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
    return child


def swap_mutation_tsp(ind):
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]


@app.route('/api/tsp', methods=['POST'])
def tsp_api():
    try:
        data = request.get_json() or {}
        pop_size = int(data.get('pop_size', 100))
        generations = int(data.get('generations', 500))
        pc = float(data.get('pc', 0.9))
        pm = float(data.get('pm', 0.2))
        elite_size = int(data.get('elite_size', 1))

        # safety caps
        pop_size = max(4, min(pop_size, 500))
        generations = max(1, min(generations, 2000))
        elite_size = max(0, min(elite_size, pop_size // 2))

        n = len(cities_tsp)
        pop = initial_population_tsp(pop_size, n)
        best = min(pop, key=lambda ind: route_distance_tsp(ind))
        best_dist = route_distance_tsp(best)
        history = []

        for g in range(generations):
            pop = sorted(pop, key=lambda ind: route_distance_tsp(ind))
            if route_distance_tsp(pop[0]) < best_dist:
                best = pop[0][:]
                best_dist = route_distance_tsp(best)

            new_pop = []
            if elite_size > 0:
                new_pop.extend(pop[:elite_size])

            while len(new_pop) < pop_size:
                p1 = tournament_selection_tsp(pop, 5)
                p2 = tournament_selection_tsp(pop, 5)
                child = ordered_crossover_tsp(p1, p2) if random.random() < pc else p1[:]
                if random.random() < pm:
                    swap_mutation_tsp(child)
                new_pop.append(child)

            pop = new_pop
            history.append({'generation': g + 1, 'best_distance': float(best_dist)})

        best_route = [cities_tsp[i] for i in best]
        return jsonify({'success': True, 'best_route': best_route, 'best_distance': float(best_dist), 'history': history[-10:]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Soft Computing Backend Server berjalan",
        "endpoints": ["/api/fuzzy", "/api/ann", "/api/genetic", "/api/genetic2", "/api/tsp"]
    })


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("Soft Computing Backend Server")
    print("=" * 60)
    print("Server berjalan di: http://localhost:5000")
    print("\nEndpoint tersedia:")
    print("  GET  /              - Halaman Utama")
    print("  GET  /fuzzy         - Halaman Fuzzy Logic")
    print("  GET  /ann           - Halaman ANN")
    print("  GET  /genetic       - Halaman Genetic Algorithm")
    print("  GET  /genetic2      - Halaman Genetic Algorithm 2 (Knapsack)")
    print("  GET  /tsp           - Halaman TSP (GA demo)")
    print("  POST /api/fuzzy     - Demo Logika Fuzzy")
    print("  POST /api/ann       - Demo Jaringan Saraf Tiruan")
    print("  POST /api/genetic   - Demo Algoritma Genetika")
    print("  POST /api/genetic2  - Demo Knapsack GA")
    print("  POST /api/tsp       - Demo TSP GA")
    print("  GET  /api/health    - Health check")
    print("\nTekan Ctrl+C untuk menghentikan server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)