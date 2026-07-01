# does .format break if not found?
mystr = "Hello, {name}!"
print(mystr.format(name="Alice"))
print(mystr.format(tasdf="Bob"))