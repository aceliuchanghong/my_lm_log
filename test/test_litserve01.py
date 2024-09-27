import litserve as ls


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model1 = lambda x: x ** 2
        self.model2 = lambda x: x ** 3

    def decode_request(self, request):
        return request["input"], request["image_url"]

    def predict(self, x):
        input, image_url = x
        squared = self.model1(input)
        cubed = self.model2(input)
        output = squared + cubed
        print(image_url+"00")
        return {"output": output}

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="gpu")
    server.run(port=8000)
