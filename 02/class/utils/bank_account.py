class BankAccount:
    def __init__(self, name):
        self.name = name
        self.balance = 0
        self.interest_rate = 0.01

    def deposit(self, num):
        self.balance += num

    def withdraw(self, num):
        self.balance -= num

    def get_balance(self):
        return self.balance

    def set_interest_rate(self, num):
        self.interest_rate = num

    def apply_interest(self):
        self.balance *= (self.interest_rate + 1)
