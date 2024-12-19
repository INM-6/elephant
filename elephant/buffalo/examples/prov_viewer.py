import sys
from elephant.buffalo.serialization import BuffaloProvDocument


def visualize(filename, output_file=None):
    prov_document = BuffaloProvDocument.read_records(filename)
    prov_document.plot(filename=output_file, show_element_attributes=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("You must inform the source file")
    visualize(sys.argv[1])
