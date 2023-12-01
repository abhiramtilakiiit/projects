[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/uO3FBJhb)
# Project Meanius - by Team DISS

Web development team project for 2nd semester spring 2023 ISS course.

## Team members

- Ankith Pai (@ankithpai) (2022113012)
- Abhiram Tilak (@abhiramtilakiiit) (2022113011)

Group name (DISS)

## Documentation

This is the documentation for the project 'Meanius' - a music review website.

## General Usage

As of right now there is only support for local viewing of webpages.

Follow the instructions below to get all the webpages up and running:

- Clone the repository for the webpage.
- Make sure all the folders including fonts and images are inside the cloned directory.
- Use a browser, preferably firefox or any browser using 'Gecko' browser engine.
- Make sure your browser supports 'websites displaying their own fonts'.
- Preferably the browser should have 'auto-hide' scrollbars option turned off for a better user experience.
- Now that your system is set up you should be able to view all the webpages as intended.

Instructions to navigate through the website:

- Open `src/index.html` in your browser which should take you to the home page.
- From here use the navigation bar to navigate through the webpages.
- Once you are in the list of Artists page, you can click on an artists taking you to the artist-albums page.
- This page consists of a brief description of the artist and the list of albums and his/her/their/its top songs.
- Clicking on any of these albums in this page should take you to the album-songs page.
- This page consists of a brief description of the album and the list of all songs in the album.
- Each song in the list consists of details like name, song-length.
- There are other accessible websites like the about page and the faq page.
- There is also a contribute page which takes you to our github.
- The footer in each of these pages also has the list of all webpages needed. which you can use for easy navigation.
- The footer also consists of credits which is a list of all resources used to make this website

## General Structure of the codebase:

```
.
├── ASSUMPTIONS.md
├── README.md
├── favicon.ico
├── features
│   ├── artist-list.html
│   ├── home-test.html
│   └── sample-artist.html
├── fonts
│   ├── (list)
│   ├── (of)
│   ├── (fonts)
├── images
│   ├── Songs
│   │   ├── (Top 3 Songs)
│   │   └── (Artist names)
│   │       ├── (Top 3)
│   │       ├── (for each)
│   │       └── (artist)
|   ├── album_covers
│   │   └── (Artist names)
│   │       └── (Their albums)
│   ├── artist
│   │   ├── (profile pictures of all)
│   │   ├── (artists and)
│   │   ├── (and their)
│   │   └── (picture for banner)
│   ├── (other images of general backgrounds)
│   ├── (banners, textures)
│   └── (and logos used around the website)
├── src
│   ├── about.html
│   ├── artists
│   │   ├── Y (Y - artist index)
│   │   |   ├── album-X.html (X - album index)
│   │   │   └── index.html
│   │   └── index.html
│   ├── css
│   │   ├── albums.css
│   │   ├── common.css
│   │   ├── footer.css
│   │   ├── home.css
│   │   └── navbar.css
│   ├── faq.html
│   └── index.html
└── webpage_content
    ├── About.txt
    ├── Faq.txt
    ├── album-description.txt
    ├── artist-descriptions.txt
    ├── cherry_picks.txt
    └── credits.txt

```

## Assumptions page

The assumptions page consists of all the assumptions that were made deviating or not mentioned in the instructions.

## The `src` dir

- Contains the source code for the entire project
- At the root of this dir we have HTML pages for "Home", "About" and "FAQ"

  #### Subdir `css`

  - All css code lives here, split into files by category

  #### Subdir `artists`

  - `index.html` is the "Artists" page.
  - Each artist has a subdir within this, identified with a unique number
  - Such a subdir has an `index.html` page which is the "Artist Albums" page, and many
    other pages for "Album Songs" that are of the form `album (x).html`

### The `images` dir

This consists of images for all the songs, albums and artists, and other backgrounds and logos used in the website:

- The artist page consists of the profile picture (small) and the banner picture (large) for all the artists.
- The albums page consists of many artists directories in each of it are all the album covers.
- The Songs page against consists of many directories in each of it are the covers of top 3 songs.
- Other images include a few banners images we used, 3 home page images, 1 texture for headings
  background, a background for the whole webpage and finally the logos.

### The `fonts` dir

- This directory consists of fonts in `.ttf` (true type fonts) downloaded from websites like nerdfonts.com
  fonts.google.com and made sure the licenses are agreed upon.
- It also consists of `fonts.txt` which contains a guide of all the font placements.

### The `features` dir

- This is for testing purposes

### The `webpage_content` dir

- This consists of all the content filled in the webpage in text files.
- It has information like albums and artists descriptions (including year of release),
  all the song names for each album, their length etc.
