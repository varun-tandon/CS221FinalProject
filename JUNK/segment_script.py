import re

def main():
	#testing script
	with open('movie_scripts/thor.html') as file:
		script = file.readlines()
		script = "".join(script)
		print(repr(script))
		dialogue = segment(script)
		for entry in dialogue:
			print(repr(entry))
			print()


def segment(x: str):
	""" given a string that contains the script, this method finds all the dialogue """
	#dialogue_regex = "(?:([A-Z]+ *[A-Z]+)\n).*?(?=$|([A-Z]+ *[A-Z]+)\n)"
	#*** IN PROGRESS ***
	dialogue_regex = r"[A-Z]+.*\n(\s)+(\w|\n)+\n\n"
	reg = r"\s*([A-Z]+)\n(.*?)\s*(.*?)(\n)(?!\s)"
	d_list = re.findall(reg, x, flags=re.DOTALL)
	return d_list


if __name__ == '__main__':
	main()