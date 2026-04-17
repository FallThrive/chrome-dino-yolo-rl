import time
from pynput.keyboard import Key, Controller
from typing import Optional, Set


class KeyboardController:
    JUMP_DURATION = 0.5
    DUCK_DURATION = 0.1
    
    def __init__(self):
        self.keyboard = Controller()
        self.key_release_queue: dict = {}
        self._current_action: Optional[str] = None
    
    def press_jump(self):
        if Key.space not in self.key_release_queue:
            self.keyboard.press(Key.space)
            self.key_release_queue[Key.space] = time.time() + self.JUMP_DURATION
            self._current_action = 'up'
    
    def press_duck(self):
        if Key.down not in self.key_release_queue:
            self.keyboard.press(Key.down)
            self.key_release_queue[Key.down] = time.time() + self.DUCK_DURATION
            self._current_action = 'down'
    
    def release_all(self):
        for key in list(self.key_release_queue.keys()):
            self.keyboard.release(key)
        self.key_release_queue.clear()
        self._current_action = None
    
    def press_enter(self):
        self.keyboard.press(Key.enter)
        self.keyboard.release(Key.enter)
    
    def press_space(self):
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
    
    def update(self):
        current_time = time.time()
        keys_to_release = []
        for key, release_time in self.key_release_queue.items():
            if current_time >= release_time:
                self.keyboard.release(key)
                keys_to_release.append(key)
        for key in keys_to_release:
            del self.key_release_queue[key]
            if key == Key.space and self._current_action == 'up':
                self._current_action = None
            elif key == Key.down and self._current_action == 'down':
                self._current_action = None
    
    def execute_action(self, action: int):
        if action == 0:
            pass
        elif action == 1:
            self.press_jump()
        elif action == 2:
            self.press_duck()
    
    def get_pressed_keys(self) -> Set[Key]:
        return set(self.key_release_queue.keys())
    
    def is_key_pressed(self, key: Key) -> bool:
        return key in self.key_release_queue
    
    @property
    def current_action(self) -> Optional[str]:
        return self._current_action
