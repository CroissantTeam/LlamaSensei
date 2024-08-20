import json


class TranscriptLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self, simple_output=False):
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print("The file was not found")
            return None
        except json.JSONDecodeError:
            print("The file does not contain valid JSON")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

        if simple_output:
            return self.data

        return self._process_data()

    def _process_data(self):
        if not self.data:
            return None

        paragraphs = self.data["results"]["channels"][0]["alternatives"][0][
            "paragraphs"
        ]["paragraphs"]
        doc = []
        for paragraph in paragraphs:
            text = " ".join([sentence["text"] for sentence in paragraph["sentences"]])
            start = paragraph["start"]
            end = paragraph["end"]
            doc.append((text, start, end))
        return doc

    def get_metadata(self):
        if self.data:
            return self.data.get("metadata")
        return None


# Example usage:
if __name__ == "__main__":
    loader = TranscriptLoader()
    result = loader.load_data()
    if result:
        print(f"Loaded {len(result)} paragraphs")
        # Print the first paragraph as an example
        if result:
            print("First paragraph:")
            print(f"Text: {result[0][0]}")
            print(f"Start time: {result[0][1]}")
            print(f"End time: {result[0][2]}")

    # To get full data
    full_data = loader.load_data(simple_output=False)
    if full_data:
        print("Full data keys:", full_data.keys())

    # To get metadata
    metadata = loader.get_metadata()
    if metadata:
        print("Metadata:", metadata)
