import sys
import os
import itertools

from pydoc import ErrorDuringImport, importfile
import inspect
from palantiri import PlotHandler


def generate_docs(module):
    try:
        sys.path.append(os.getcwd())
        # Attempt import
        mod = importfile(module)
        if mod is None:
            print("Module not found")

        # Module imported correctly, let's create the docs
        return get_markdown(mod)
    except ErrorDuringImport:
        print("Error while trying to import " + module)


def get_markdown(module):
    output = [module.__name__, "***\n"]
    output.extend(get_classes(module))
    return "".join(output)


def get_classes(item):

    def sort_classes(class_list):
        for passnum in range(len(class_list) - 1, 0, -1):

            for i in range(passnum):

                if issubclass(class_list[i], class_list[i + 1]):
                    temp = class_list[i]
                    class_list[i] = class_list[i + 1]
                    class_list[i + 1] = temp

        return class_list

    classes_dict = dict()
    output = list()
    for cl in inspect.getmembers(item, inspect.isclass):
        if cl[0] != "__class__" and not cl[0].startswith("_"):
            classes_dict[cl[1]] = ["## "+cl[0]+'\n']
            output.append("## "+cl[0]+'\n')
            # Get the docstring
            output.append(inspect.getdoc(cl[1])+'\n')
            classes_dict[cl[1]].append(inspect.getdoc(cl[1])+'\n')
            # Get the functions
            output.extend(get_functions(cl[1]))
            classes_dict[cl[1]].extend(get_functions(cl[1]))

    c_list = list(classes_dict.keys())
    c_list = sort_classes(c_list)

    if len(c_list) == 1:
        return classes_dict[c_list[0]]
    c_list.remove(PlotHandler)
    output_updated = list(itertools.chain(*[classes_dict[i] for i in c_list]))
    return output_updated


def get_functions(item):

    output = list()
    for func in inspect.getmembers(item, inspect.isfunction):

        output.append("### "+func[0]+str(inspect.signature(func[1]))+'\n')

        # Get the docstring
        output.append(inspect.getdoc(func[1])+'\n')
    return output


if __name__ == '__main__':

    directory_in_str = '/home/wolfenfeld/Development/palantiri/palantiri'

    directory = os.fsencode(directory_in_str)

    results = list()

    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        if filename.endswith(".py") and filename != "__init__.py":
            result = generate_docs(os.path.join(directory_in_str, filename))
            results.append(result)
            with open('/home/wolfenfeld/Development/temp/{0}.md'.format(result.split("***")[0]), 'w') as the_file:
                the_file.write(result)

        else:
            continue

    print(results)
