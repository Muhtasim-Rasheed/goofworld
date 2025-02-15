# Goofworld!

goofworld! (v1.1.3)
This is a funny little test world where I learn Rust and macroquad and some other stuff.

# Fix fixity fix fix

- Performance upgrade to `random_string(usize)` function
- Models are included in the binary now using `include_bytes!` macro
- Learned that I have to triangulate my models before exporting them to .obj (Quads and NGons are not supported)
- Ability to change texture of models and tint them with colors
- Now using a struct instead of a function to load models
- Added a new model: A chair

# What you can do right now

- Walk around with WASD
- Shift to sneak / crouch (very minecrafty)
- Q to sprint
- CTRL to take back control of the mouse
- Mouse to look around
- Space to jump
- ESC to quit

# What GAMEOBJECTS can do

- [Glungus](https://glungussy-wussy.fandom.com/wiki/Glungus) can follow you (funy)
- There's a text screaming "GOOFWORLD" at x0, y10, z0
- A green cube that moves from y1 to y4 and back
- holycrackers.png is avaiable
- planecat.png is also avaiable
- Two models: A table and a chair (around x4, y0, z-5)
