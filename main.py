import simpy as sp

def car(env, name, bcs, driving_time, charge_duration):
    while True:
        yield env.timeout(driving_time)
        print('%s arriving at %d' % (name, env.now))
        with bcs.request() as req:
            yield req

            # Charge the battery
            print('%s starting to charge at %s' % (name, env.now))
            yield env.timeout(charge_duration)
            print('%s leaving the bcs at %s' % (name, env.now))

if __name__ == "__main__":
    env = sp.Environment()
    bcs = sp.Resource(env, capacity=2)
    for i in range(4):
        env.process(car(env, "Car %d" % i, bcs, i * 2.2, 5.2))
        
    env.run(until=61.02)