from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, value

class SprintPlanner:
    def __init__(self):
        # Incizalizē pamatdatus no tabulām
        self.sprints = list(range(1, 7))
        self.capacity = {1:25, 2:20, 3:20, 4:30, 5:25, 6:15}

        self.user_stories = [1,2,5,6,7,8,9,10,13,14,15,16,17,18,19,20,56,57,58,59,60]
        
        self.utility = {1:80, 2:85, 5:80, 6:45, 7:50, 8:25, 9:35, 10:30,
                       13:75, 14:70, 15:60, 16:65, 17:25, 18:20, 19:10, 20:25,
                       56:75, 57:65, 58:60, 59:35, 60:30}
        
        self.story_points = {1:9, 2:7, 5:7, 6:5, 7:6, 8:4, 9:5, 10:4,
                             13:10, 14:7, 15:6, 16:8, 17:4, 18:5, 19:6, 20:7,
                             56:8, 57:6, 58:9, 59:5, 60:4}
        
        self.risks = {1:1.7, 2:1.7, 5:1.3, 6:1.0, 7:1.3, 8:1.0, 9:1.0, 10:1.0,
                      13:1.7, 14:1.3, 15:1.7, 16:1.7, 17:1.3, 18:2.0, 19:1.0, 20:1.0,
                      56:2.0, 57:2.0, 58:1.0, 59:1.3, 60:1.0}
        
        # Korelācijas
        self.correlating_stories = [1, 2, 18, 56]
        self.correlations = {
            1: {'affinity': 0.3, 'correlated': [2], 'count': 1},
            2: {'affinity': 0.3, 'correlated': [1], 'count': 1},
            18: {'affinity': 0.5, 'correlated': [56], 'count': 1},
            56: {'affinity': 0.5, 'correlated': [18], 'count': 1}
        }
        
        # Pirmsnorises
        self.dependant_stories = [7, 8, 10, 13, 15, 17, 18, 57]
        self.dependencies = {
            7: {'depending_on': [5,6], 'count': 1},  # OR tapēc "1" nevis "2"
            8: {'depending_on': [7], 'count': 1},
            10: {'depending_on': [9], 'count': 1},
            13: {'depending_on': [16], 'count': 1},
            15: {'depending_on': [2,16], 'count': 2},
            17: {'depending_on': [15], 'count': 1},
            18: {'depending_on': [17], 'count': 1},
            57: {'depending_on': [56], 'count': 1}
        }
        
        # Otimizācijas modeļa izveide
        self.model = LpProblem("Sprint_Planning", LpMaximize)
        self.X = None
        self.Y = None
        
    def create_model(self):
        # Izveidojam x un y (i - sprints, j - stāsts)
        self.X = LpVariable.dicts("X", 
            [(i,j) for i in self.sprints for j in self.user_stories],
            cat = LpBinary)
        
        self.Y = LpVariable.dicts("Y",
            [(i,j) for i in self.sprints for j in self.correlating_stories],
            lowBound = 0, upBound = None, cat = 'Integer')
        
        # Mērķa funkcija
        base_value = lpSum(
            self.utility[j] * self.risks[j] * self.X[(i,j)]
            for i in self.sprints for j in self.user_stories
        )
        
        correlation_value = lpSum(
            (self.utility[j] * self.correlations[j]['affinity'] * self.Y[(i,j)]) / 
            max(1, self.correlations[j]['count'])
            for i in self.sprints for j in self.correlating_stories
        )

        self.model += base_value + correlation_value

        # Bāzes nosacījumi
        # Sprinta ietilpība
        for i in self.sprints:
            self.model += (
                lpSum(self.story_points[j] * self.X[(i,j)] for j in self.user_stories)
                <= self.capacity[i],
                f"Capacity_Sprint_{i}"
            )
        
        # Stāsta iekļaušana
        for j in self.user_stories:
            self.model += (
                lpSum(self.X[(i,j)] for i in self.sprints) == 1,
                f"Story_inclusion_{j}"
            )
        
        # Nosacījumi pirmsnorisēm
        for j in self.dependant_stories:
            dep = self.dependencies[j]
            for i in self.sprints:
                if dep['count'] == 1:
                    if len(dep['depending_on']) == 1:  # AND
                        self.model += (
                            self.X[(i,j)] <= lpSum(self.X[(k,d)] 
                            for k in range(1, i) for d in dep['depending_on']),
                            f"DependencyAND_Sprint{i}_Story{j}"
                        )
                    else: # OR gadījums
                        self.model += (
                            self.X[(i,j)] <= lpSum(self.X[(k,d)] 
                            for k in range(1, i) for d in dep['depending_on']),
                            f"DependencyOR_Sprint{i}_Story{j}"
                        )
                else:
                    self.model += (
                        self.X[(i,j)] * dep['count'] <= 
                        lpSum(self.X[(k,d)] for k in range(1, i) 
                        for d in dep['depending_on']),
                        f"DependencyMultipleAND_Sprint{i}_Story{j}"
                    )
        
        # Nosacījumi korelācijai
        for j in self.correlating_stories:
            for i in self.sprints:
                self.model += (
                    self.Y[(i,j)] <= lpSum(self.X[(i,k)] 
                    for k in self.correlations[j]['correlated']),
                    f"Correlation1_Sprint{i}_Story{j}"
                )
                
                self.model += (
                    self.Y[(i,j)] <= self.correlations[j]['count'] * self.X[(i,j)],
                    f"Correlation2_Sprint{i}_Story{j}"
                )
    

    def solve(self):
        self.model.solve()
        print(f"Status: {LpStatus[self.model.status]}")
        print(f"Objective Value: {value(self.model.objective)}")

    def print_solution(self):
        print("\nOptimal Sprint Assignment:")
        for i in self.sprints:
            print(f"\nSprint {i} (Capacity: {self.capacity[i]} points):")
            total_points = 0
            for j in self.user_stories:
                if value(self.X[(i,j)]) == 1:
                    print(f"  Story R.{j:02d} - Points: {self.story_points[j]}, " +
                        f"Utility: {self.utility[j]}, Risk: {self.risks[j]}")
                    total_points += self.story_points[j]
            print(f"  Total points used: {total_points}/{self.capacity[i]}")

# Main - metožu palaišana
if __name__ == "__main__":
    planner = SprintPlanner()
    planner.create_model()
    planner.solve()
    planner.print_solution()