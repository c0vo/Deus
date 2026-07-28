import asyncio
import websockets

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Test a complex query
        query = "Analyze the macro impact of the recent fed rate cuts on tech stocks."
        print(f"Sending: {query}")
        await websocket.send(query)
        
        while True:
            try:
                response = await websocket.recv()
                print(f"Received: {response}")
                # stop if done
                import json
                try:
                    data = json.loads(response)
                    if data.get("type") == "done":
                        break
                except:
                    pass
            except Exception as e:
                print("Error:", e)
                break

asyncio.run(test_chat())
