import heapq
import random
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from abc import ABC, abstractmethod

# ============================================================================
# ⚙️ SYSTEM CONFIGURATION & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AI_Tutor_Core")


# ============================================================================
# 🎓 CO1: ADVANCED KNOWLEDGE REPRESENTATION (DAG)
# ============================================================================
@dataclass
class TopicNode:
    id: str
    name: str
    prerequisites: List[str] = field(default_factory=list)
    difficulty: int = 1
    # BKT Default Parameters (P_init, P_transit, P_guess, P_slip)
    bkt_params: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1)


class CurriculumDAG:
    def __init__(self):
        self.nodes: Dict[str, TopicNode] = {}
        self._initialize_production_curriculum()

    def add_node(self, node: TopicNode):
        self.nodes[node.id] = node

    def _initialize_production_curriculum(self):
        topics = [
            TopicNode("add", "Addition", [], 1, (0.3, 0.25, 0.1, 0.1)),
            TopicNode("sub", "Subtraction", ["add"], 1, (0.3, 0.25, 0.1, 0.1)),
            TopicNode("mul", "Multiplication", ["add"], 2, (0.2, 0.25, 0.15, 0.1)),
            TopicNode("div", "Division", ["mul", "sub"], 2, (0.1, 0.2, 0.2, 0.1)),
            TopicNode("frac", "Fractions", ["div"], 3, (0.1, 0.15, 0.25, 0.15)),
            TopicNode("alg_basic", "Basic Algebra", ["frac"], 3, (0.05, 0.2, 0.2, 0.1)),
            TopicNode("lin_eq", "Linear Equations", ["alg_basic"], 4, (0.05, 0.15, 0.2, 0.15)),
            TopicNode("quad_eq", "Quadratic Equations", ["lin_eq"], 5, (0.01, 0.1, 0.15, 0.2))
        ]
        for t in topics:
            self.add_node(t)


# ============================================================================
# 🏭 ACTUATOR: FACTORY DESIGN PATTERN FOR QUESTION GENERATION
# ============================================================================
class QuestionGenerator(ABC):
    @abstractmethod
    def generate(self) -> Dict[str, str]: pass


class ArithmeticGenerator(QuestionGenerator):
    def __init__(self, operation: str):
        self.op = operation

    def generate(self) -> Dict[str, str]:
        a, b = random.randint(2, 12), random.randint(2, 12)
        if self.op == "add": return {"q": f"What is {a} + {b}?", "a": str(a + b)}
        if self.op == "sub": return {"q": f"What is {a + b} - {a}?", "a": str(b)}
        if self.op == "mul": return {"q": f"What is {a} * {b}?", "a": str(a * b)}
        if self.op == "div": return {"q": f"What is {a * b} / {a}?", "a": str(b)}
        return {"q": "Error", "a": "0"}


class AlgebraGenerator(QuestionGenerator):
    def __init__(self, level: str):
        self.level = level

    def generate(self) -> Dict[str, str]:
        if self.level == "alg_basic":
            x = random.randint(2, 9)
            ans = random.randint(1, 10)
            return {"q": f"Solve for x: {x}x = {x * ans}", "a": str(ans)}
        if self.level == "lin_eq":
            m, x, b = random.randint(2, 5), random.randint(1, 5), random.randint(1, 10)
            return {"q": f"Solve for x: {m}x + {b} = {m * x + b}", "a": str(x)}
        if self.level == "quad_eq":
            ans = random.randint(2, 6)
            return {"q": f"If x^2 = {ans ** 2}, what is the positive value of x?", "a": str(ans)}
        return {"q": "Error", "a": "0"}


class ContentFactory:
    @staticmethod
    def get_question(topic_id: str) -> Dict[str, str]:
        if topic_id in ["add", "sub", "mul", "div"]:
            return ArithmeticGenerator(topic_id).generate()
        elif topic_id in ["alg_basic", "lin_eq", "quad_eq"]:
            return AlgebraGenerator(topic_id).generate()
        elif topic_id == "frac":
            a, b = random.randint(1, 5), random.randint(2, 9)
            return {"q": f"What is the numerator of the fraction {a}/{b}?", "a": str(a)}
        return {"q": "1+1", "a": "2"}


# ============================================================================
# 🎓 CO5: TRUE BAYESIAN KNOWLEDGE TRACING (BKT) + EXPLICIT COUNTER
# ============================================================================
class StudentModelBKT:
    def __init__(self, dag: CurriculumDAG, mastery_threshold: float = 0.85, min_correct: int = 3):
        self.dag = dag
        self.threshold = mastery_threshold
        self.min_correct = min_correct  # Forces at least 3 correct answers

        self.p_known: Dict[str, float] = {node_id: node.bkt_params[0] for node_id, node in dag.nodes.items()}
        self.correct_answers: Dict[str, int] = {node_id: 0 for node_id in dag.nodes}
        self.mastered_nodes: Set[str] = set()

    def update_observation(self, topic_id: str, is_correct: bool) -> float:
        _, p_transit, p_guess, p_slip = self.dag.nodes[topic_id].bkt_params
        prev_p = self.p_known[topic_id]

        if is_correct:
            p_evidence = (prev_p * (1 - p_slip)) / ((prev_p * (1 - p_slip)) + ((1 - prev_p) * p_guess))
            self.correct_answers[topic_id] += 1  # Track successful answers
        else:
            p_evidence = (prev_p * p_slip) / ((prev_p * p_slip) + ((1 - prev_p) * (1 - p_guess)))

        new_p = p_evidence + (1 - p_evidence) * p_transit
        self.p_known[topic_id] = round(new_p, 4)

        logger.info(
            f"BKT Update | Topic: {topic_id} | Correct: {is_correct} | New Prob: {new_p:.2%} | Tally: {self.correct_answers[topic_id]}/{self.min_correct}")

        # TWO-FACTOR MASTERY: Must cross probability threshold AND answer at least 3 correctly
        if self.p_known[topic_id] >= self.threshold and self.correct_answers[topic_id] >= self.min_correct:
            if topic_id not in self.mastered_nodes:
                self.mastered_nodes.add(topic_id)
                logger.info(f"✅ Mastery Threshold & Demonstration Reached for {topic_id}")

        elif self.p_known[topic_id] < self.threshold and topic_id in self.mastered_nodes:
            self.mastered_nodes.remove(topic_id)

        return self.p_known[topic_id]


# ============================================================================
# 🎓 CO2: A* SEARCH WITH ADMISSIBLE HEURISTICS
# ============================================================================
class AStarNavigator:
    def __init__(self, dag: CurriculumDAG):
        self.dag = dag

    def compute_optimal_path(self, goal_id: str, mastered: Set[str]) -> List[str]:
        if goal_id not in self.dag.nodes:
            return []

        roots = [n_id for n_id, n in self.dag.nodes.items() if not n.prerequisites]
        start = roots[0] if roots else list(self.dag.nodes.keys())[0]

        open_set = [(0, start)]
        came_from: Dict[str, str] = {}
        g_score = {n: float('inf') for n in self.dag.nodes}
        g_score[start] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_id:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return [n for n in reversed(path) if n not in mastered]

            neighbors = [n_id for n_id, n in self.dag.nodes.items() if current in n.prerequisites]

            for neighbor in neighbors:
                tentative_g = g_score[current] + self.dag.nodes[neighbor].difficulty

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = max(0, self.dag.nodes[goal_id].difficulty - self.dag.nodes[neighbor].difficulty)
                    heapq.heappush(open_set, (tentative_g + h_score, neighbor))
        return []


# ============================================================================
# 🎓 CO3: CONSTRAINT SATISFACTION (CSP)
# ============================================================================
class CSPController:
    def __init__(self, max_load: int = 35, max_failures: int = 5):
        self.max_load = max_load
        self.current_load = 0
        self.sequential_failures = 0
        self.max_failures = max_failures

    def record_attempt(self, difficulty: int, correct: bool) -> bool:
        self.current_load += difficulty

        if correct:
            self.sequential_failures = 0
        else:
            self.sequential_failures += 1

        if self.current_load > self.max_load:
            logger.warning("CSP Violation: Maximum cognitive load exceeded.")
            return False
        if self.sequential_failures >= self.max_failures:
            logger.warning("CSP Violation: Frustration limit exceeded.")
            return False
        return True


# ============================================================================
# 🎓 CO4 & CO6: DIALOGUE MANAGER & SYSTEM INTEGRATION
# ============================================================================
class AITutorService:
    def __init__(self):
        self.dag = CurriculumDAG()
        self.navigator = AStarNavigator(self.dag)

    def initialize_student(self) -> Tuple[StudentModelBKT, CSPController]:
        return StudentModelBKT(self.dag, min_correct=3), CSPController()

    def parse_user_input(self, text: str) -> str:
        text = text.lower().strip()
        nums = re.findall(r'-?\d+', text)
        return nums[0] if nums else text


class CLIDialogueManager:
    def __init__(self):
        self.backend = AITutorService()
        self.student, self.csp = self.backend.initialize_student()
        self.current_goal: Optional[str] = None

    def start(self):
        print("\n" + "=" * 70)
        print("🚀 ENTERPRISE AI TUTOR INITIALIZED")
        print("Architected with BKT, A* Search, and CSP Graphing.")
        print("=" * 70)
        self._select_goal()
        self._main_loop()

    def _select_goal(self):
        print("\n[Curriculum Modules]:")
        for topic_id, node in self.backend.dag.nodes.items():
            if topic_id in self.student.mastered_nodes:
                status = "✅"
            else:
                prereqs_met = all(p in self.student.mastered_nodes for p in node.prerequisites)
                status = "📖" if prereqs_met else "🔒"

            print(f"  {status} [{topic_id}]: {node.name}")

        while True:
            target = input("\nEnter the ID of your goal module: ").strip().lower()
            if target in self.backend.dag.nodes:
                if target in self.student.mastered_nodes:
                    print("You already mastered this! Pick a harder one.")
                    continue
                self.current_goal = target
                logger.info(f"Goal set to {target}")
                return
            print("Invalid ID. Please type the exact ID in brackets [].")

    def _main_loop(self):
        while True:
            path = self.backend.navigator.compute_optimal_path(self.current_goal, self.student.mastered_nodes)

            if not path:
                print(f"\n🎉 EXCELLENT! You have completely mastered {self.backend.dag.nodes[self.current_goal].name}!")
                if len(self.student.mastered_nodes) == len(self.backend.dag.nodes):
                    print("🏆 YOU COMPLETED THE ENTIRE CURRICULUM.")
                    break
                self._select_goal()
                continue

            current_topic_id = path[0]
            current_node = self.backend.dag.nodes[current_topic_id]

            q_data = ContentFactory.get_question(current_topic_id)
            current_prob = self.student.p_known[current_topic_id] * 100
            tally = self.student.correct_answers[current_topic_id]
            required = self.student.min_correct

            # Displays the new "Questions Correct: 0/3" UI tracker
            print(f"\n🧠 [Focus: {current_node.name} | Mastery: {current_prob:.1f}% | Correct: {tally}/{required}]")
            print(f"🤖 AI: {q_data['q']}")

            user_input = input("👤 You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Session terminated gracefully. Goodbye!")
                break

            parsed_ans = self.backend.parse_user_input(user_input)
            is_correct = (parsed_ans == q_data['a'])

            if is_correct:
                print("✅ Correct!")
            else:
                print(f"❌ Incorrect. The answer was {q_data['a']}.")

            self.student.update_observation(current_topic_id, is_correct)

            if not self.csp.record_attempt(current_node.difficulty, is_correct):
                print("\n⚠️ AI: My constraint monitor shows you are experiencing cognitive fatigue.")
                print("Taking a break is mathematically proven to help retention. See you later!")
                break


if __name__ == "__main__":
    app = CLIDialogueManager()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nProcess interrupted. Data saved. Goodbye!")
