import h5py

path = "/Users/anabellafalk/cs6758/LIBERO/libero/datasets/libero_object/"
file = "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"

with h5py.File(path + file, "r") as f:
    print("Top-level keys:", list(f.keys()))

    demos = f["data"]
    print("Number of demonstrations:", len(demos))

    demo = demos["demo_0"]
    print("Demo keys:", list(demo.keys()))

    obs = demo["obs"]
    print("Observation keys:", list(obs.keys()))

    print("Actions shape:", demo["actions"].shape)
    print("Agentview image shape:", obs["agentview_rgb"].shape)