
import pygame
import random
import math
import time
import matplotlib.pyplot as p

class ChaseEnvironment:
    def __init__(self):
        self.screen_width = 1200
        self.screen_height = 800
        self.border_width = 10
        self.chaser_size = 20
        self.target_size = 30
        self.chaser_color = (0, 0, 255)
        self.target_color = (255, 0, 0)
        self.chaser_speed = 8
        self.target_speed = 4
        self.chaser_acceleration = 0.5
        self.max_chaser_speed = 50
        self.max_target_speed = 45
        self.target_accleration = 0.01
        self.target_change_direction_threshold = 50
        self.num_obstacles = 5
        self.max_obstacles = 20
        self.obstacle_size = 45
        self.obstacles = []

        self.chaser_positions = [[self.screen_width//2, self.screen_height//2]]
        self.reset()
        
    def reset_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            x = random.randint(self.border_width + 50, self.screen_width - self.border_width - self.obstacle_size - 50)
            y = random.randint(self.border_width + 50, self.screen_height - self.border_width - self.obstacle_size - 50)
            self.obstacles.append(pygame.Rect(x, y, self.obstacle_size, self.obstacle_size))


    def reset(self):
        self.chaser_x = self.screen_width // 2
        self.chaser_y = self.screen_height // 2
        self.target_x = random.randint(self.border_width + 300, self.screen_width - self.border_width - self.target_size - 300)
        self.target_y = random.randint(self.border_width + 200, self.screen_height - self.border_width - self.target_size - 200)
        self.target_dx = random.choice([-1, 1]) * self.target_speed
        self.target_dy = random.choice([-1, 1]) * self.target_speed
        self.reward = 0
        self.steps = 0
        self.score = 0
        self.chaser_positions = [[self.screen_width//2, self.screen_height//2]]
        #Uncomment the below for the Obstacles Challenge
        # self.reset_obstacles()
        return self.get_state()

    def chaser_action(self, action):
        # Modified to support 8 directional movement
        # 0: Up, 1: Down, 2: Left, 3: Right
        # 4: Down-Right, 5: Up-Right, 6: Down-Left, 7: Up-Left
        
        diagonal_speed = self.chaser_speed / math.sqrt(2)  # Normalized diagonal speed
        
        if action == 0:  # Up
            self.chaser_y -= self.chaser_speed
        elif action == 1:  # Down
            self.chaser_y += self.chaser_speed
        elif action == 2:  # Left
            self.chaser_x -= self.chaser_speed
        elif action == 3:  # Right
            self.chaser_x += self.chaser_speed
        elif action == 4:  # Down-Right
            self.chaser_x += diagonal_speed
            self.chaser_y += diagonal_speed
        elif action == 5:  # Up-Right
            self.chaser_x += diagonal_speed
            self.chaser_y -= diagonal_speed
        elif action == 6:  # Down-Left
            self.chaser_x -= diagonal_speed
            self.chaser_y += diagonal_speed
        elif action == 7:  # Up-Left
            self.chaser_x -= diagonal_speed
            self.chaser_y -= diagonal_speed
        elif action == 8:  # Increase speed
            self.chaser_speed = min(self.chaser_speed + self.chaser_acceleration, self.max_chaser_speed)
        elif action == 9:  # Decrease speed
            self.chaser_speed = max(5, self.chaser_speed - self.chaser_acceleration)
            
        # Ensure chaser stays within boundaries
        self.chaser_x = max(self.border_width, min(self.chaser_x, self.screen_width - self.border_width - self.chaser_size))
        self.chaser_y = max(self.border_width, min(self.chaser_y, self.screen_height - self.border_width - self.chaser_size))

        # Uncomment this for the Obstacles Challenge
        # chaser_rect = pygame.Rect(self.chaser_x, self.chaser_y, self.chaser_size, self.chaser_size)
        # for obstacle in self.obstacles:
        #     if chaser_rect.colliderect(obstacle):
        #         self.chaser_x, self.chaser_y = prev_x, prev_y  # Revert movement
        #         break


    def target_movement(self):
        self.target_x += self.target_dx
        self.target_y += self.target_dy
        if (self.target_x <= self.border_width + 10 or
            self.target_x >= self.screen_width - self.border_width - self.target_size - 10):
            self.target_dx *= -1
            self.target_dx += random.uniform(-0.5, 0.5)
        if (self.target_y <= self.border_width + 10 or
            self.target_y >= self.screen_height - self.border_width - self.target_size - 10):
            self.target_dy *= -1
            self.target_dy += random.uniform(-0.5, 0.5)
        self.target_x = max(self.border_width, min(self.target_x, self.screen_width - self.border_width - self.target_size))
        self.target_y = max(self.border_width, min(self.target_y, self.screen_height - self.border_width - self.target_size))
        if self.steps % self.target_change_direction_threshold == 0:
            self.target_dx = random.choice([-self.target_speed, self.target_speed]) * self.target_speed + random.uniform(-1, 1)
            self.target_dy = random.choice([-self.target_speed, self.target_speed]) * self.target_speed + random.uniform(-1, 1)
            magnitude = math.sqrt(self.target_dx**2 + self.target_dy**2)
            self.target_dx = (self.target_dx / magnitude) * self.target_speed
            self.target_dy = (self.target_dy / magnitude) * self.target_speed
        self.steps += 1
        distance = self.get_distance()
        if distance < 200:
            dx = self.chaser_x - self.target_x
            dy = self.chaser_y - self.target_y
            mag = math.sqrt(dx**2 + dy**2)
            if mag > 0:
                dx /= mag
                dy /= mag
                self.target_dx -= dx * 0.5
                self.target_dy -= dy * 0.5
                mag = math.sqrt(self.target_dx**2 + self.target_dy**2)
                if mag > 0:
                    self.target_dx = (self.target_dx / mag) * self.target_speed
                    self.target_dy = (self.target_dy / mag) * self.target_speed

    def get_distance(self):
        return math.sqrt((self.target_x - self.chaser_x)**2 + (self.target_y - self.chaser_y)**2)

    def get_reward(self):
        current_distance = self.get_distance()
        if current_distance <= self.chaser_size + self.target_size:
            self.reward += 10
            self.score += 1
            return 10
        elif (self.chaser_x <= self.border_width or
              self.chaser_x >= self.screen_width - self.border_width - self.chaser_size or
              self.chaser_y <= self.border_width or
              self.chaser_y >= self.screen_height - self.border_width - self.chaser_size):
            self.reward -= 5
            self.chaser_x = self.screen_width//2
            self.chaser_y = self.screen_height//2
            return -5
        else:
            if len(self.chaser_positions) >= 2:
                prev_distance = math.sqrt((self.target_x - self.chaser_positions[0][0])**2 +
                                         (self.target_y - self.chaser_positions[0][1])**2)
                if current_distance < prev_distance:
                    return 0.5
                else:
                    return -0.2
            else:
                return 0

    def step(self, action):
        self.chaser_action(action)
        self.target_movement()
        self.chaser_positions.append([self.chaser_x, self.chaser_y])
        if len(self.chaser_positions) > 5:
            self.chaser_positions.pop(0)
        reward = self.get_reward()
        state = self.get_state()
        done = False
        if reward == 10:
            #Uncomment the below for the Obstacles Challenge
            # if self.score % 5 == 0:
            #     additional = random.randint(1, 3)
            #     self.num_obstacles = min(self.num_obstacles + additional, self.max_obstacles)
            
            # self.reset_obstacles()

            done = True
            self.target_x = random.randint(self.border_width + 100, self.screen_width - self.border_width - self.target_size - 100)
            self.target_y = random.randint(self.border_width + 100, self.screen_height - self.border_width - self.target_size - 100)
            self.target_dx = random.choice([-1, 1]) * self.target_speed
            self.target_dy = random.choice([-1, 1]) * self.target_speed
            self.target_speed = min(self.target_speed + 0.2, self.max_target_speed)
        self.target_speed = min(self.target_speed + 0.05,self.max_target_speed)
        return state, reward, done

    def get_state(self):
        distance = self.get_distance()
        return {
            'target_x': self.target_x,
            'target_y': self.target_y,
            'chaser_x': self.chaser_x,
            'chaser_y': self.chaser_y,
            'distance': distance,
            'chaser_speed': self.chaser_speed,
            'target_speed': self.target_speed,
            'target_dx': self.target_dx,
            'target_dy': self.target_dy,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height
        }

    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Chase Environment")
        return screen

    def update_screen(self, screen):
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, self.chaser_color, (self.chaser_x, self.chaser_y, self.chaser_size, self.chaser_size))
        pygame.draw.rect(screen, self.target_color, (self.target_x, self.target_y, self.target_size, self.target_size))
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, self.screen_width, self.screen_height), self.border_width)
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        speed_text = font.render(f'Chaser Speed: {self.chaser_speed:.1f}', True, (0, 0, 0))
        target_speed_text = font.render(f'Target Speed: {self.target_speed:.1f}', True, (0, 0, 0))
        screen.blit(score_text, (20, 20))
        screen.blit(speed_text, (20, 50))
        screen.blit(target_speed_text, (20, 80))
        #Uncomment the below for the Obstacles Challenge
        # for obstacle in self.obstacles:
        #     pygame.draw.rect(screen, (100, 100, 100), obstacle)

        pygame.display.flip()



    def close_window(self):
        pygame.quit()

import pygame
import random
import math
import time
import numpy as np
import pickle
import os
from collections import deque

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Action space: 0-7 for direction, 8 for speed up, 9 for slow down
        self.action_space = 10
        
        # State discretization parameters
        self.grid_size = 20  # Grid cells for discretizing position
        self.angle_bins = 8  # Number of bins for angle discretization
        self.distance_bins = 10  # Number of bins for distance discretization
        self.speed_bins = 5  # Number of bins for speed discretization
        
        # Initialize Q-table
        self.q_table = {}
        
        # Performance tracking
        self.captures = 0
        self.last_capture_time = time.time()
        self.capture_times = []
        self.rewards_history = []
        self.epsilon_history = []
        
        # Load existing model if available
        self.model_path = "q_learning_chase_model.pkl"
        if os.path.exists(self.model_path):
            self.load_model()
        
    def discretize_state(self, state):
        """
        Convert continuous state to discrete state for Q-table lookup
        """
        # Extract key information
        cx, cy = state['chaser_x'], state['chaser_y']
        tx, ty = state['target_x'], state['target_y']
        tdx, tdy = state['target_dx'], state['target_dy']
        distance = state['distance']
        chaser_speed = state['chaser_speed']
        target_speed = state['target_speed']
        
        # Grid position of chaser (relative to screen)
        grid_x = min(int(cx / (self.env.screen_width / self.grid_size)), self.grid_size - 1)
        grid_y = min(int(cy / (self.env.screen_height / self.grid_size)), self.grid_size - 1)
        
        # Relative position of target to chaser
        dx = tx - cx
        dy = ty - cy
        
        # Convert to angle and distance
        angle = math.atan2(dy, dx)
        # Normalize angle to 0-2π range
        if angle < 0:
            angle += 2 * math.pi
        
        # Discretize angle into bins
        angle_bin = min(int(angle / (2 * math.pi / self.angle_bins)), self.angle_bins - 1)
        
        # Discretize distance into bins (max distance is diagonal of screen)
        max_distance = math.sqrt(self.env.screen_width**2 + self.env.screen_height**2)
        distance_bin = min(int(distance / (max_distance / self.distance_bins)), self.distance_bins - 1)
        
        # Discretize target direction
        target_angle = math.atan2(tdy, tdx)
        if target_angle < 0:
            target_angle += 2 * math.pi
        target_angle_bin = min(int(target_angle / (2 * math.pi / self.angle_bins)), self.angle_bins - 1)
        
        # Discretize speeds
        chaser_speed_bin = min(int(chaser_speed / (self.env.max_chaser_speed / self.speed_bins)), self.speed_bins - 1)
        target_speed_bin = min(int(target_speed / (self.env.max_target_speed / self.speed_bins)), self.speed_bins - 1)
        
        # Combine into discrete state tuple
        # Using only the most important features to keep Q-table manageable
        discrete_state = (angle_bin, distance_bin, target_angle_bin, chaser_speed_bin)
        
        return discrete_state
    
    def get_q_value(self, state, action):
        """Get Q value for state-action pair, initialize if not exists"""
        discrete_state = self.discretize_state(state)
        
        if discrete_state not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[discrete_state] = np.random.uniform(0, 0.1, self.action_space)
            
        return self.q_table[discrete_state][action]
    
    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        """
        if training and random.random() < self.exploration_rate:
            # Exploration: random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploitation: best known action
            discrete_state = self.discretize_state(state)
            
            if discrete_state not in self.q_table:
                # Initialize with small random values
                self.q_table[discrete_state] = np.random.uniform(0, 0.1, self.action_space)
                
            # Return action with highest Q-value
            return np.argmax(self.q_table[discrete_state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule
        """
        # Get current Q value
        current_q = self.get_q_value(state, action)
        
        # Get max Q value for next state
        discrete_next_state = self.discretize_state(next_state)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.random.uniform(0, 0.1, self.action_space)
        
        max_next_q = np.max(self.q_table[discrete_next_state])
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[self.discretize_state(state)][action] = new_q
    
    def update_exploration_rate(self):
        """
        Decrease exploration rate over time
        """
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
        self.epsilon_history.append(self.exploration_rate)
    
    def save_model(self):
        """
        Save the Q-table to disk
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """
        Load the Q-table from disk
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
            # Set to minimum exploration once model is loaded
            self.exploration_rate = self.min_exploration
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def train(self, episodes=500, max_steps=1000, visualize=True):
        """
        Train the agent for the specified number of episodes
        """
        if visualize:
            screen = self.env.render()
        else:
            screen = None
            
        total_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                steps += 1
                
                # Update state
                state = next_state
                
                # Check if we captured the target
                if done:
                    self.captures += 1
                    current_time = time.time()
                    capture_time = current_time - self.last_capture_time
                    self.capture_times.append(capture_time)
                    self.last_capture_time = current_time
                    
                # Visualize if requested
                if visualize and screen:
                    self.env.update_screen(screen)
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.env.close_window()
                            return
                    
                    # Slow down visualization to see what's happening
                    pygame.time.delay(10)
            
            # Update exploration rate after each episode
            self.update_exploration_rate()
            
            # Store episode stats
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            self.rewards_history.append(total_reward)
            
            # Print episode info
            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:]) if total_rewards else 0
                print(f"Episode {episode}: Steps={steps}, Reward={total_reward:.2f}, "
                      f"Avg Reward={avg_reward:.2f}, Epsilon={self.exploration_rate:.4f}, "
                      f"Captures={self.captures}")
                
            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                self.save_model()
                
        # Final save
        self.save_model()
        
        if visualize and screen:
            self.env.close_window()
            
        return total_rewards, episode_lengths
    
    def run(self, num_steps=3000, visualize=True):
        """
        Run the trained agent in evaluation mode
        """
        if visualize:
            screen = self.env.render()
        else:
            screen = None
            
        state = self.env.reset()
        total_reward = 0
        start_time = time.time()
        self.captures = 0
        self.last_capture_time = start_time
        self.capture_times = []  # Reset capture times list
        
        for step in range(num_steps):
            # Get action (no exploration)
            action = self.get_action(state, training=False)
            
            # Take action
            next_state, reward, done = self.env.step(action)
            
            total_reward += reward
            state = next_state
            
            # Check for captures
            if done:
                current_time = time.time()
                self.captures += 1
                capture_interval = current_time - self.last_capture_time
                self.capture_times.append(capture_interval)
                self.last_capture_time = current_time
                print(f"Capture {self.captures}! Time since last capture: {capture_interval:.2f}s")
            
            if visualize and screen:
                self.env.update_screen(screen)
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.env.close_window()
                        return total_reward, 0  # Return early with 0 CPS
                
                # Calculate performance metrics periodically
                if step % 100 == 0:
                    elapsed_time = time.time() - start_time
                    captures_per_second = self.captures / elapsed_time if elapsed_time > 0 else 0
                    
                    # Display performance metrics in console periodically
                    if self.captures > 0:
                        print(f"Step {step}/{num_steps} | Total captures: {self.captures}")
                        print(f"Elapsed time: {elapsed_time:.2f}s")
                        print(f"Current CPS: {captures_per_second:.2f}")
        
        # Calculate final statistics
        total_elapsed_time = time.time() - start_time
        captures_per_second = self.captures / total_elapsed_time if total_elapsed_time > 0 else 0
        
        if visualize and screen:
            self.env.close_window()
            
        return total_reward, captures_per_second


def main():
    # Initialize environment
    env = ChaseEnvironment()
    
    # Initialize Q-learning agent
    agent = QLearningAgent(env)
    
    # Check if we want to train or run
    train_mode = input("Train the agent? (y/n): ").lower() == 'y'
    
    if train_mode:
        # Get training parameters
        episodes = int(input("Number of episodes (default 200): ") or 200)
        visualize = input("Visualize training? (y/n): ").lower() == 'y'
        
        print(f"Starting training for {episodes} episodes...")
        rewards, steps = agent.train(episodes=episodes, visualize=visualize)
        
       
        
        # Print training statistics
        print("\n--- Training Statistics ---")
        print(f"Total captures: {agent.captures}")
        print(f"Average reward per episode: {np.mean(rewards):.2f}")
        print(f"Average episode length: {np.mean(steps):.2f}")
        
        # Save the trained model
        agent.save_model()
        
        # Ask if we should run the trained agent
        run_after_train = input("Run the trained agent? (y/n): ").lower() == 'y'
        if not run_after_train:
            return
    
    # Run the agent in evaluation mode
    print("Running autonomous agent...")
    print("Press Ctrl+C to exit")
    
    try:
        # Reset agent captures counter for clean evaluation
        agent.captures = 0
        
        # Run for 3000 steps and get back both reward and CPS
        total_reward, captures_per_second = agent.run(3000)
        
        # Show final statistics
        if len(agent.capture_times) > 0:
            avg_time_between_captures = sum(agent.capture_times) / len(agent.capture_times)
        else:
            avg_time_between_captures = 0
        
        print("\n--- Final Statistics ---")
        print(f"Total captures: {agent.captures}")
        print(f"Total reward: {total_reward:.2f}")
        if agent.captures > 0:
            print(f"Average time between captures: {avg_time_between_captures:.2f} seconds")
        print(f"Capture rate: {captures_per_second:.2f} captures per second")
        
    except KeyboardInterrupt:
        print("\nStopping the simulation...")
    finally:
        env.close_window()

if __name__ == "__main__":
    main()