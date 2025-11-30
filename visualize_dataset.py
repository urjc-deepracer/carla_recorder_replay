#-------------------------------------------------
#----Visualize dataset generated from replay.py---
#-------------------------------------------------

import os
import time
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the dataset generated from replay.py"
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help="Base directory where dataset is located"
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


# Plots for throttle, steer, speed 
def render_plot(df, index, window=50):
    start = max(0, index - window)
    data_slice = df[start:index + 1]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.tight_layout(pad=2.0)

    timestamps = data_slice['timestamp']

    axs[0, 0].plot(timestamps, data_slice['throttle'], color='green')
    axs[0, 0].set_title("Throttle [0,1]")
    axs[0, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 0].set_ylim(0.0, 1.1)

    axs[0, 1].plot(timestamps, data_slice['steer'], color='blue')
    axs[0, 1].set_title("Steer [-1,1]")
    axs[0, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 1].set_ylim(-1.1, 1.1)

    axs[1, 0].plot(timestamps, data_slice['brake'], color='red')
    axs[1, 0].set_title("Brake [0,1]")
    axs[1, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[1, 0].set_ylim(0.0, 1.0)

    axs[1, 1].plot(timestamps, data_slice['speed'], color='orange')
    axs[1, 1].set_title("Speed (m/s)")
    axs[1, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[1, 1].set_ylim(0, 35)
   


    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plt.close(fig)
    return surf


def main():
    args = parse_args()
    BASE_PATH = args.base_path

    CSV_PATH = os.path.join(BASE_PATH, "dataset.csv")
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] Unable to find dataset.csv en: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    pygame.init()
    screen = pygame.display.set_mode((1900, 1000))
    pygame.display.set_caption("Visualize Dataset DeepRacer")

    font = pygame.font.SysFont(None, 26)
    font_big = pygame.font.SysFont(None, 48)
    clock = pygame.time.Clock()

    index = 0
    running = True

    while running and index < len(df):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        row = df.loc[index]
        plot_surface = render_plot(df, index)

        screen.fill((20, 20, 20))
        screen.blit(plot_surface, (800, 100)) 

        # Header
        txt = f"Frame: {index} | Timestamp: {int(row['timestamp'])}"
        text_surf = font.render(txt, True, (255, 255, 255))
        screen.blit(text_surf, (50, 10))

        # Load images 
        rgb_rel  = row.iloc[0]
        mask_rel = row.iloc[1]

        imagen_rgb_path  = os.path.join(BASE_PATH, rgb_rel.lstrip("/"))
        imagen_mask_path = os.path.join(BASE_PATH, mask_rel.lstrip("/"))

        if os.path.isfile(imagen_rgb_path):
            img_rgb = pygame.image.load(imagen_rgb_path).convert_alpha()
            screen.blit(img_rgb, (0, 40))
        else:
            warn = font.render(f"RGB not found: {imagen_rgb_path}", True, (255, 100, 100))
            screen.blit(warn, (0, 40))

        if os.path.isfile(imagen_mask_path):
            img_mask = pygame.image.load(imagen_mask_path).convert_alpha()
            screen.blit(img_mask, (0, 500))
        else:
            warn = font.render(f"Mask not found: {imagen_mask_path}", True, (255, 100, 100))
            screen.blit(warn, (0, 500))

        pygame.display.flip()
        time.sleep(1/1000)
        index += 1

    pygame.quit()


if __name__ == "__main__":
    main()
