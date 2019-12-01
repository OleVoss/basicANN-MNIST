import tkinter as tk

root = tk.Tk()
root.geometry("500x500")

for r in range(28):
	for c in range(28):
		tk.Button(root, width=1, height=1).grid(row=r, column=c)




root.mainloop()