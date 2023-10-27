import argparse

def read_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file]
            return lines
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file and print its contents.')
    parser.add_argument('file_path', type=str, help='Path to the file to be read')
    args = parser.parse_args()
    
    lines = read_file(args.file_path)