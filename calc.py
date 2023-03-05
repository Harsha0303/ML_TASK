import tkinter as tk

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Calculator")

        # Creating the input field
        self.input_field = tk.Entry(self, width=40, borderwidth=5)
        self.input_field.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Creating buttons for the calculator
        self.create_button("1", 1, 0)
        self.create_button("2", 1, 1)
        self.create_button("3", 1, 2)
        self.create_button("4", 2, 0)
        self.create_button("5", 2, 1)
        self.create_button("6", 2, 2)
        self.create_button("7", 3, 0)
        self.create_button("8", 3, 1)
        self.create_button("9", 3, 2)
        self.create_button("0", 4, 1)
        self.create_button(".", 4, 2)

        # Creating operation buttons
        self.create_button("+", 1, 3)
        self.create_button("-", 2, 3)
        self.create_button("*", 3, 3)
        self.create_button("/", 4, 3)

        self.create_button("Clear", 5, 0)
        self.create_button("=", 5, 1, columnspan=2)

    def create_button(self, text, row, column, columnspan=1):
        button = tk.Button(self, text=text, padx=40, pady=20, command=lambda: self.button_click(text))
        button.grid(row=row, column=column, columnspan=columnspan)

    def button_click(self, text):
        if text == "Clear":
            self.input_field.delete(0, tk.END)
        elif text == "=":
            self.calculate()
        else:
            self.input_field.insert(tk.END, text)

    def calculate(self):
        input_string = self.input_field.get()
        try:
            result = eval(input_string)
            self.input_field.delete(0, tk.END)
            self.input_field.insert(0, result)
        except:
            self.input_field.delete(0, tk.END)
            self.input_field.insert(0, "Error")

calculator = Calculator()
calculator.mainloop()
