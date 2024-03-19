import pandas as pd

from collections import deque

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter


class Graph:
    def __init__(self):
        self.graph = {}
        self.edges = []
        self.bridges = []
        self.time = 0
        self.timer = 0
        self.time_in = {}
        self.min_time = {}
        self.st = []
        self.components = []
        self.parent = {}
        self.rank = {}
        self.geolocator = Nominatim(user_agent="geoapiExercises")
        self.countries = {
            1: "Тирана Албания",
            2: "Андорра-ла-Велла Андорра",
            3: "Ереван Армения",
            4: "Вена Австрия",
            5: "Баку Азербайджан",
            6: "Минск Беларусь",
            7: "Брюссель Бельгия",
            8: "Сараево Босния и Герцеговина",
            9: "София Болгария",
            10: "Загреб Хорватия",
            11: "Никосия Кипр",
            12: "Прага Чехия",
            13: "Копенгаген Дания",
            14: "Таллин Эстония",
            15: "Хельсинки Финляндия",
            16: "Париж Франция",
            17: "Берлин Германия",
            18: "Тбилиси Грузия",
            19: "Афины Греция",
            20: "Будапешт Венгрия",
            21: "Рейкьявик Исландия",
            22: "Дублин Ирландия",
            23: "Рим Италия",
            24: "Рига Латвия",
            25: "Вадуц Лихтенштейн",
            26: "Вильнюс Литва",
            27: "Люксембург Люксембург",
            28: "Валлетта Мальта",
            29: "Кишинев Молдова",
            30: "Монако Монако",
            31: "Подгорица Черногория",
            32: "Амстердам Нидерланды",
            33: "Скопье Северная Македония",
            34: "Осло Норвегия",
            35: "Варшава Польша",
            36: "Лиссабон Португалия",
            37: "Бухарест Румыния",
            38: "Москва Россия",
            39: "Сан-Марино Сан-Марино",
            40: "Белград Сербия",
            41: "Братислава Словакия",
            42: "Любляна Словения",
            43: "Мадрид Испания",
            44: "Стокгольм Швеция",
            45: "Берн Швейцария",
            46: "Анкара Турция",
            47: "Киев",
            48: "Лондон Великобритания",
            49: "Ватикан Ватикан",
        }
        self.geocode = RateLimiter(
            self.geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10
        )

    def add_edge(self, v1, v2):
        """Add an edge between vertices v1 and v2."""

        if v1 not in self.graph:
            self.graph[v1] = []
        if v2 not in self.graph:
            self.graph[v2] = []

        self.graph[v1].append(v2)
        self.graph[v2].append(v1)

    def calculate_degree_of_vertices(self):
        """Calculate the degree of each vertex in the largest connected component of the graph."""
        largest_component = self.get_largest_component()
        degrees = {vertex: 0 for vertex in largest_component}

        for vertex in largest_component:
            if vertex in self.graph:  # Проверяем, есть ли вершина в графе
                degrees[vertex] = len(
                    [
                        neighbor
                        for neighbor in self.graph[vertex]
                        if neighbor in largest_component
                    ]
                )

        return degrees

    def dfs(self, vertex, visited, component):
        """Perform a depth-first search starting from vertex"""
        visited.add(vertex)
        component.append(vertex)
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                self.dfs(neighbor, visited, component)

    def bfs(self, start_vertex):
        """Perform a breadth-first search starting from start_vertex"""
        distances = {vertex: float("inf") for vertex in self.graph}
        queue = deque([start_vertex])
        distances[start_vertex] = 0

        while queue:
            current_vertex = queue.popleft()
            for neighbor in self.graph[current_vertex]:
                if distances[neighbor] == float("inf"):
                    distances[neighbor] = distances[current_vertex] + 1
                    queue.append(neighbor)

        return distances

    def find_connected_components(self):
        """Find all connected components in the graph"""
        visited = set()
        components = []

        all_vertices = set(range(1, max(self.graph) + 1))

        for vertex in all_vertices:
            if vertex not in visited:
                component = []
                if vertex not in self.graph or not self.graph[vertex]:
                    components.append([vertex])
                    visited.add(vertex)
                else:
                    self.dfs(vertex, visited, component)
                    components.append(sorted(component))

        components.sort(key=len, reverse=True)

        return components

    def get_largest_component(self):
        """Get the largest connected component in the graph"""
        components = self.find_connected_components()
        largest_component = max(components, key=len)

        return set(largest_component)

    def calculate_eccentricity(self):
        """Calculate the eccentricity of each vertex in the graph"""
        largest_component = self.get_largest_component()
        eccentricity = {}

        for vertex in largest_component:
            shortest_paths = self.bfs(vertex)
            reachable_distances = [
                dist
                for dest, dist in shortest_paths.items()
                if dist != float("inf") and dest in largest_component
            ]
            eccentricity[vertex] = max(reachable_distances)

        return eccentricity

    def calculate_radius_diameter_center(self):
        """Calculate the radius, diameter, and center of the graph"""
        eccentricity = self.calculate_eccentricity()

        radius = min(eccentricity.values())
        diameter = max(eccentricity.values())
        center = [vertex for vertex, ecc in eccentricity.items() if ecc == radius]

        return radius, diameter, center

    def generate_distance_matrix(self):
        """Generate the distance matrix of the graph"""
        largest_component = self.get_largest_component()
        vertices = sorted(largest_component)
        matrix = []

        for v in vertices:
            distances = self.bfs(v)
            row = [
                distances[dest]
                if distances[dest] != float("inf") and dest in largest_component
                else -1
                for dest in vertices
            ]
            matrix.append(row)
        df = pd.DataFrame(matrix, index=vertices, columns=vertices)

        return df

    def bron_kerbosch(self, R, P, X, cliques):
        """Bron-Kerbosch algorithm for finding all maximal cliques in the graph"""
        if len(P) == 0 and len(X) == 0:
            cliques.append(R)

            return

        for vertex in P.copy():
            newR = R.union({vertex})
            newP = P.intersection(self.graph[vertex])
            newX = X.intersection(self.graph[vertex])

            self.bron_kerbosch(newR, newP, newX, cliques)
            P.remove(vertex)
            X.add(vertex)

    def find_max_clique_bron_kerbosch(self):
        """Find the maximum clique in the graph using the Bron-Kerbosch algorithm"""
        largest_component = self.get_largest_component()

        P = set(largest_component)
        R = set()
        X = set()

        all_cliques = []
        self.bron_kerbosch(R, P, X, all_cliques)

        max_size = max(len(clique) for clique in all_cliques)
        max_cliques = [clique for clique in all_cliques if len(clique) == max_size]

        return max_cliques

    def color_vertices_dsatur(self):
        """Color the vertices of the graph using the DSatur algorithm"""
        largest_component = list(self.get_largest_component())
        n = len(largest_component)

        color = {vertex: -1 for vertex in largest_component}
        saturation = {vertex: 0 for vertex in largest_component}
        degree = {vertex: len(self.graph[vertex]) for vertex in largest_component}

        vertex = max(degree, key=degree.get)
        color[vertex] = 0
        assigned = set([vertex])

        while len(assigned) < n:
            for v in largest_component:
                if v not in assigned:
                    saturation[v] = len(
                        set(
                            color[neigh]
                            for neigh in self.graph[v]
                            if neigh in assigned and color[neigh] != -1
                        )
                    )

            next_vertex = max(
                set(largest_component) - assigned,
                key=lambda x: (saturation[x], degree[x]),
            )

            used_colors = set(
                color[neigh] for neigh in self.graph[next_vertex] if neigh in assigned
            )
            color[next_vertex] = next(i for i in range(n) if i not in used_colors)
            assigned.add(next_vertex)

        return color

    def extract_bcc_vertices(self, u, v, st, bcc):
        """Extract a vertices biconnected component from the stack"""
        bcc_set = set()

        while st:
            x, y = st.pop()
            bcc_set.add(x)
            bcc_set.add(y)
            if (x, y) == (u, v) or (y, x) == (u, v):
                break

        if bcc_set:
            bcc.append(bcc_set)

    def tarjan_util_vertices(self, u, low, disc, visited, parent, ap, st, bcc):
        """Utility function for Tarjan's algorithm for finding biconnected components and articulation points"""
        children = 0
        visited[u] = True
        disc[u] = self.time
        low[u] = self.time
        self.time += 1

        for v in self.graph.get(u, []):
            if not visited[v]:
                parent[v] = u
                children += 1
                st.append((u, v))
                self.tarjan_util_vertices(v, low, disc, visited, parent, ap, st, bcc)

                low[u] = min(low[u], low[v])

                if parent[u] is None and children > 1:
                    ap.add(u)

                if parent[u] is not None and low[v] >= disc[u]:
                    ap.add(u)
                    self.extract_bcc_vertices(u, v, st, bcc)

            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
                if disc[v] < disc[u]:
                    st.append((u, v))

    def tarjan_vertices(self):
        """Find vertices biconnected components and cut vertices using Tarjan's algorithm"""
        visited = {v: False for v in self.graph}
        disc = {v: float("inf") for v in self.graph}
        low = {v: float("inf") for v in self.graph}
        parent = {v: None for v in self.graph}
        ap = set()
        st = []
        bcc = []

        for u in self.graph:
            if not visited[u]:
                self.tarjan_util_vertices(u, low, disc, visited, parent, ap, st, bcc)
                bcc_set = set()
                while st:
                    x, y = st.pop()
                    bcc_set.add(x)
                    bcc_set.add(y)
                if bcc_set:
                    bcc.append(bcc_set)

        return bcc, ap

    def dfs_edge_biconnected_components(self, v, parent=-1):
        """Depth-first search for finding edge biconnected components and bridges"""
        self.time_in[v] = self.min_time[v] = self.timer + 1
        self.timer += 1
        self.st.append(v)

        for u in self.graph[v]:
            if u == parent:
                continue
            if u not in self.time_in:
                cur_size = len(self.st)
                min_u_time = self.dfs_edge_biconnected_components(u, v)
                self.min_time[v] = min(self.min_time[v], min_u_time)

                if min_u_time > self.time_in[v]:
                    self.bridges.append((v, u))
                    self.new_component(cur_size)
            else:
                self.min_time[v] = min(self.min_time[v], self.time_in[u])

        return self.min_time[v]

    def new_component(self, cur_size):
        """Create a new edge biconnected component"""
        component = []
        while len(self.st) != cur_size:
            component.append(self.st.pop())
        if component:
            self.components.append(component)

    def find_edge_biconnected_components(self):
        """Find edge biconnected components in the graph"""
        for v in self.graph:
            if v not in self.time_in:
                self.dfs_edge_biconnected_components(v)
                self.new_component(0)

    def make_set(self, v):
        """Make a set with a single vertex."""
        self.parent[v] = v
        self.rank[v] = 0

    def find_set(self, v):
        """Find the representative of the set that v is in."""
        if v != self.parent[v]:
            self.parent[v] = self.find_set(self.parent[v])

        return self.parent[v]

    def union_sets(self, a, b):
        """Combine two sets into one."""
        a = self.find_set(a)
        b = self.find_set(b)

        if a != b:
            if self.rank[a] < self.rank[b]:
                a, b = b, a
            self.parent[b] = a

            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1

    def add_weighted_edge_with_weight(self, u, v, weight):
        """Add a weighted edge between vertices u and v with weight"""
        self.edges.append({"u": u, "v": v, "weight": weight})

    def add_weighted_edges_for_largest_component_with_weights(self, edges_weights):
        """Add weighted edges for the largest connected component of the graph"""
        for u, v, weight in edges_weights:
            if u in self.get_largest_component() and v in self.get_largest_component():
                self.add_weighted_edge_with_weight(u, v, weight)

    def find_mst_kruskal(self):
        """Find the minimum spanning tree of the graph using Kruskal's algorithm"""
        for vertex in self.get_largest_component():
            self.make_set(vertex)

        self.edges.sort(key=lambda e: e["weight"])
        mst_weight = 0
        mst_edges = []

        for edge in self.edges:
            if self.find_set(edge["u"]) != self.find_set(edge["v"]):
                self.union_sets(edge["u"], edge["v"])
                mst_weight += edge["weight"]
                mst_edges.append(edge)

        return mst_weight, mst_edges

    def generate_prufer_code(self, mst_edges):
        """Generate the Prufer code for the minimum spanning tree"""
        adjacency_list = {}
        for u, v in mst_edges:
            if u not in adjacency_list:
                adjacency_list[u] = []
            if v not in adjacency_list:
                adjacency_list[v] = []
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)

        vertex_degree = {u: len(adjacency_list[u]) for u in adjacency_list}

        leaves = sorted([u for u in vertex_degree if vertex_degree[u] == 1])

        prufer_code = []
        while len(adjacency_list) > 2:
            leaf = leaves.pop(0)
            neighbor = adjacency_list[leaf][0]

            prufer_code.append(neighbor)

            vertex_degree[neighbor] -= 1
            adjacency_list[neighbor].remove(leaf)

            if vertex_degree[neighbor] == 1:
                leaves.append(neighbor)
                leaves.sort()

            del adjacency_list[leaf]
            del vertex_degree[leaf]

        return prufer_code

    def dfs_for_bin_code(self, vertex, visited, ans):
        """Depth-first search for generating the binary code for the minimum spanning tree"""
        visited[vertex] = True
        for neighbor in self.graph.get(vertex, []):
            if not visited[neighbor]:
                ans.append(1)
                self.dfs_for_bin_code(neighbor, visited, ans)
        ans.append(0)

    def generate_binary_code(self):
        """Generate the binary code for the minimum spanning tree"""
        visited = {vertex: False for vertex in self.graph}
        ans = []

        self.dfs_for_bin_code(1, visited, ans)

        return ans


if __name__ == "__main__":
    edges = [
        (1, 19),
        (1, 33),
        (1, 31),
        (1, 40),
        (2, 43),
        (2, 16),
        (3, 5),
        (3, 18),
        (3, 46),
        (4, 12),
        (4, 17),
        (4, 45),
        (4, 25),
        (4, 23),
        (4, 42),
        (4, 20),
        (4, 41),
        (5, 18),
        (5, 38),
        (6, 38),
        (6, 24),
        (6, 26),
        (6, 35),
        (6, 47),
        (7, 32),
        (7, 16),
        (7, 17),
        (7, 27),
        (8, 10),
        (8, 31),
        (8, 40),
        (9, 37),
        (9, 40),
        (9, 33),
        (9, 19),
        (9, 46),
        (10, 42),
        (10, 20),
        (10, 40),
        (12, 17),
        (12, 41),
        (12, 35),
        (13, 17),
        (24, 14),
        (38, 14),
        (15, 38),
        (15, 44),
        (15, 34),
        (16, 17),
        (16, 27),
        (16, 45),
        (16, 23),
        (16, 43),
        (17, 32),
        (17, 27),
        (17, 45),
        (17, 35),
        (18, 38),
        (18, 46),
        (19, 46),
        (19, 33),
        (20, 41),
        (20, 42),
        (20, 40),
        (20, 37),
        (23, 49),
        (23, 39),
        (23, 45),
        (23, 42),
        (24, 38),
        (24, 26),
        (25, 45),
        (26, 38),
        (26, 35),
        (29, 47),
        (29, 37),
        (30, 16),
        (31, 40),
        (33, 40),
        (34, 44),
        (34, 38),
        (35, 38),
        (35, 47),
        (35, 41),
        (36, 43),
        (37, 47),
        (37, 40),
        (38, 47),
        (41, 47),
        (47, 20),
        (5, 46),
        (10, 31),
        (22, 48),
    ]

    graph = Graph()
    for v1, v2 in edges:
        graph.add_edge(v1, v2)

    print("---------------------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------------------")
    print("Degree of vertices:")
    degrees_of_vertices = graph.calculate_degree_of_vertices()
    print(degrees_of_vertices)

    print("\n")

    print("Minimum degree:", min(degrees_of_vertices.values()))
    print("Maximum degree:", max(degrees_of_vertices.values()))
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Connected components:")
    connected_components = graph.find_connected_components()
    print(connected_components)

    print("\n")

    print("Largest component:")
    largest_component = graph.get_largest_component()
    print(largest_component)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Eccentricity of vertices:")
    eccentricity = graph.calculate_eccentricity()
    print(eccentricity)

    print("\n")

    radius, diameter, center = graph.calculate_radius_diameter_center()
    print("Radius:", radius)
    print("Diameter:", diameter)
    print("Center:", center)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Distance matrix:")
    distance_matrix = graph.generate_distance_matrix()
    distance_matrix.to_csv("distance_matrix.csv")
    print(distance_matrix)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Max cliques: ")
    max_clique_bk = graph.find_max_clique_bron_kerbosch()
    print(max_clique_bk)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Coloring vertices:")
    coloring = graph.color_vertices_dsatur()
    for vertex, color in sorted(coloring.items(), key=lambda x: x[0]):
        print(f"Vertex {vertex}: Color {color + 1}")
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    biconnected_components, cut_vertices = graph.tarjan_vertices()
    print("Biconnected Components:")
    for component in biconnected_components:
        print(component)

    print("\n")

    print("Cut vertices:")
    print(cut_vertices)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    graph.find_edge_biconnected_components()
    print("Edge biconnected components:")
    for component in graph.components:
        print(component)

    print("\n")

    print("Bridges:")
    print(graph.bridges)
    print("---------------------------------------------------------------------------------------------------------------")

    edges_weights = [
        (1, 19, 501.4),
        (1, 33, 153.5),
        (1, 31, 132),
        (1, 40, 391.2),
        (2, 43, 494.3),
        (2, 16, 708.3),
        (3, 5, 452.9),
        (3, 18, 170),
        (3, 46, 994.5),
        (4, 12, 252.8),
        (4, 17, 524),
        (4, 45, 685.1),
        (4, 25, 527.7),
        (4, 23, 765.5),
        (4, 42, 276.6),
        (4, 20, 214.7),
        (4, 41, 57.3),
        (5, 18, 447.6),
        (5, 38, 1930.3),
        (6, 38, 677.5),
        (6, 24, 403.6),
        (6, 26, 172.1),
        (6, 35, 476.9),
        (6, 47, 434.2),
        (7, 32, 173),
        (7, 16, 265.1),
        (7, 17, 651.2),
        (7, 27, 187.7),
        (8, 10, 289.4),
        (8, 31, 172.1),
        (8, 40, 196.9),
        (9, 37, 296.2),
        (9, 40, 329.7),
        (9, 33, 174.1),
        (9, 19, 525.5),
        (9, 46, 855.1),
        (10, 42, 117),
        (10, 20, 300),
        (10, 40, 368.2),
        (12, 17, 279.7),
        (12, 41, 292.1),
        (12, 35, 518.5),
        (13, 17, 356.8),
        (24, 14, 279.6),
        (38, 14, 870.4),
        (15, 38, 895),
        (15, 44, 397.4),
        (15, 34, 790.6),
        (16, 17, 879),
        (16, 27, 288),
        (16, 45, 436.3),
        (16, 23, 1106.6),
        (16, 43, 1052.5),
        (17, 32, 577.6),
        (17, 27, 602.4),
        (17, 45, 752.3),
        (17, 35, 519.5),
        (18, 38, 1648),
        (18, 46, 1026),
        (19, 46, 819),
        (19, 33, 487.8),
        (20, 41, 160.1),
        (20, 42, 381.7),
        (20, 40, 317.3),
        (20, 37, 644.1),
        (23, 49, 2.7),
        (23, 39, 227),
        (23, 45, 689.6),
        (23, 42, 489.5),
        (24, 38, 844.6),
        (24, 26, 262.4),
        (25, 45, 158.8),
        (26, 38, 792.8),
        (26, 35, 394),
        (29, 47, 400.7),
        (29, 37, 357.6),
        (30, 16, 690.2),
        (31, 40, 281),
        (33, 40, 323.2),
        (34, 44, 418.8),
        (34, 38, 1649.8),
        (35, 38, 1154.3),
        (35, 47, 691.6),
        (35, 41, 530.4),
        (36, 43, 503.9),
        (37, 47, 746.8),
        (37, 40, 450),
        (38, 47, 756.6),
        (41, 47, 1004.6),
        (47, 20, 901.4),
        (5, 46, 1445.4),
        (10, 31, 457.6),
    ]

    print("---------------------------------------------------------------------------------------------------------------")
    graph.add_weighted_edges_for_largest_component_with_weights(edges_weights)
    mst_weight, mst_edges = graph.find_mst_kruskal()
    mst_edges_simple_format = [(edge["u"], edge["v"]) for edge in mst_edges]
    print("MST edges:")
    print(mst_edges_simple_format)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    print("Prufer code for MST:")
    prufer_code = graph.generate_prufer_code(mst_edges_simple_format)
    print(prufer_code)
    print("---------------------------------------------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------------------------------------------")
    graph.graph = {}
    for edge in mst_edges:
        graph.add_edge(edge["u"], edge["v"])

    binary_code = graph.generate_binary_code()
    print("Binary code for MST:")
    print(binary_code)
    print("---------------------------------------------------------------------------------------------------------------")