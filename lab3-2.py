import paho.mqtt.client as mqtt
import numpy as np

# 0. Define callbacks - functions that run when events happen.

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connection returned result: " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    # client.subscribe("ece180d/test")

# The callback of the client when it disconnects.
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected Disconnect")
    else:
        print("Expected Disconnect")

# The default message callback.
# (won’t be used if only publishing, but can still exist)
def on_message(client, userdata, message):
    print('Received message: "' + str(message.payload.decode()) +
          '" on topic "' + message.topic +
          '" with QoS ' + str(message.qos))

# 1. Create a client instance.
client = mqtt.Client()

# Add callbacks to client.
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

# 2. Connect to a broker using one of the connect*() functions.
client.connect_async('test.mosquitto.org')

# 3. Call one of the loop*() functions to maintain network traffic flow with the broker.
client.loop_start()

# 4. Use subscribe() to subscribe to a topic and receive messages.
# (Optional: uncomment to test receiving)
# client.subscribe("ece180d/test")

# 5. Use publish() to publish messages to the broker.
# Payload must be a string, bytearray, int, float or None.
for i in range(10):
    value = float(np.random.random(1))
    client.publish('ece180d/test', value, qos=1)
    print(f"Published: {value}")

# 6. Use disconnect() to disconnect from the broker.
client.loop_stop()
client.disconnect()
