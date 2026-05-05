---
title: Coping with Diacritics
slug: diacritics
order: 2
year: '2004'
source_url: https://www.accesstoinsight.org/lib/diacritics.html
license: CC-BY
---

# Writing without an alphabet

Pali is a phonetic language with no written alphabet of its own. Students of the language have therefore relied on their own native alphabets to read and write Pali, ever since the 1st century BCE, when Sri Lankan scribes first recorded the Tipitaka in the Sinhala alphabet. But when Europeans began to take an interest in South Asian languages in the 19th century, they quickly discovered that their own roman alphabet was no match for the wide range of phonemes (sounds) present in South Asian languages. European scholars thus began representing the more problematic Pali phonemes by augmenting the roman alphabet with a system of letter-pairs and diacritics, including the macron (horizontal bar), dot-over, dot-under, and tilde:

Until well into the mid-20th century, Pali typefaces using these characters were used almost exclusively by specialty book publishers; a scholar's day-to-day duties of transcribing, translating, and editing had to be laboriously carried out with typewriter, pen, and a steady hand with which to apply the diacritics. Unfortunately, the first personal computers failed to address the typographic challenge of diacritics, as they were designed around a very limited character set (ASCII) that was only barely able to accommodate the upper- and lower-case roman letters, ten digits, and a modest sprinkling of punctuation marks. The extended-ASCII set, which soon followed, offered a suite of additional special symbols, including many required for northern- and eastern- European alphabets. But still no macrons or dot-unders. In the absence of a universally accepted computer representation of non-ASCII characters, students of non-European languages were left to invent their own stopgap methods. These range from giving ordinary punctuation marks double-duty as stand-ins for diacritics, to designing special diacritic fonts (all of which are incompatible with each other), and everything in between.

# Evaluating the methods

A good written phonetic representation of Pali — indeed, of any language — using one's native alphabet as a starting point should aspire to each of the following ideals:

  1. It should be **_readable_** by a wide audience. It should introduce a minimum of special characters that are not already present in the alphabet. It is better to modify an existing letter with a small diacritic than to introduce an entirely new character that may look like an alien squiggle to the uninitiated. A newcomer to Pali, upon seeing a _t_ with a dot-under, should be able to guess immediately that the letter stands for some variant of a _t_ sound.
  2. It should be **_phonetically precise_**. The written text should precisely and accurately capture the phonetic content. Each phoneme (sound) should be unambiguously represented by a unique letter or combination of letters.
  3. It should be **_easy to type_**. Writing Pali should not be a cumbersome exercise in keyboard gymnastics. Typing an _a-_ macron should not call for a long series of keystrokes (e.g., Alt-Ctl-Shift-Esc-a).
  4. It should be **_portable_**. If you hand me a book — or send me a text file by e-mail — it should appear to me exactly as it did to you. I should be able to sound out the text phonetically exactly as you intended.

No single method simultaneously realizes all of these goals; no single method is "best" Your method of choice therefore depends both on your particular needs (e.g., Do you demand phonetic precision? Are you printing a book or dashing off a quick e-mail?) and on the typing, printing, and computing resources at your disposal (e.g., Do you have a Pali font? Does your text-editing app support Unicode?).

In what follows I've singled out some of the more common strategies that Pali students have used in recent decades, running the gamut from ignoring diacritics altogether to using Unicode fonts. I evaluate the success of each strategy in achieving the above-mentioned goals, to help you decide which method best suits your needs.

## Method 1. Ignore the diacritics

This is certainly the simplest method. But the cost of that simplicity is heavy: the irretrievable loss of crucial pronunciation details. This is the method often used here at Access to Insight. (I should add that I do make use of the palatal nasal _ñ_ because it is so easy to implement using HTML and because it is contained in the extended-ASCII character set found on practically everyone's computer nowadays.)

Examples: |  panatipata veramani sikkha-padam samadiyami [1]   
(HTML: panatipata veramani sikkha-padam samadiyami) itihidam ayasmato kondaññassa, añña-kondañño'tveva namam, ahositi [2]   
(HTML: itihidam ayasmato konda&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-konda&ntilde;&ntilde;o'tveva namam, ahositi).  
---|---  
Readability: | Excellent  
Phonetics: | Poor  
Ease of use: | Excellent  
Portability: | Excellent  
Overall: | Fair. Its phonetic imprecision renders it next to useless in substantive discussions of Pali grammar  
Uses: | Informal correspondence, email. OK for low-budget print projects that don't require linguistic precision.  
  
## Method 2. Use capital letters

Capitalized letters represent letters with an accompanying diacritic. The method is simple, but it has ambiguities: how, for example, would you distinguish between the palatal and guttural _n_ (n with a dot-under, and n with a dot-over)?

Examples: |  pANAtipAtA veramaNI sikkhA-padaM samAdiyAmi   
(HTML: pANAtipAtA veramaNI sikkhA-padaM samAdiyAmi) itihidaM Ayasmato koNDaññassa, añña-koNDañño'tveva nAmaM, ahosIti   
(HTML: itihidaM Ayasmato koNDa&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-koNDa&ntilde;&ntilde;o'tveva nAmaM, ahosIti)  
---|---  
Readability: | Poor. The ever-shifting case is disturbing. When caps appear at the end of a word it looks like mirror writing.  
Phonetics: | Fair. The palatal _n_ and guttural _n_ are indistinguishable.  
Ease of use: | Good. It may take time to get used to the shift key's new significance.  
Portability: | Excellent  
Overall: | Fair. The manic appearance of caps at random points is hard to bear.  
Uses: | Informal correspondence, email. Not suitable for print.  
  
## Method 3. The Velthuis scheme: double the vowels, punctuate the consonants

In the [Velthuis scheme](<../abbrev.html#velthuis>) two basic rules are observed:

  1. Long vowels (those usually typeset with a macron (bar) above them) are doubled: _aa ii uu_
  2. For consonants, the diacritic mark precedes the letter it affects. Thus, the retroflex (cerebral) consonants (usually typeset with a dot underneath) are: _.t .th .d .dh .n .l_. The pure nasal (niggahiita) m, also typeset with a dot underneath, is _.m_. The guttural nasal (_n_ with a dot above) is represented as _"n ._ The palatal nasal _(n_ with a tilde) is _~n._

Of the plain-ASCII methods, this one is the most precise, as it carefully preserves the significance of each special character. To the uninitiated, however, the sight of all those doubled vowels and misplaced periods is utterly bewildering, perhaps leaving them to wonder if someone's keyboard is broken.

Examples: |  paa.naatipaataa verama.nii sikkhaa-pada.m samaadiyaami   
(HTML: paa.naatipaataa verama.nii sikkhaa-pada.m samaadiyaami) itihida.m aayasmato ko.n.daññassa, añña-ko.n.dañño'tveva naama.m, ahosiiti   
(HTML: itihida.m aayasmato ko.n.da&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-ko.n.da&ntilde;&ntilde;o'tveva naama.m, ahosiiti)  
---|---  
Readability: | Fair. Text looks like it has been sprinkled with typos.  
Phonetics: | Excellent.  
Ease of use: | Good. Requires learning the dual significance of the period and double-quote keys.  
Portability: | Excellent  
Overall: | Good.  
Uses: | Formal scholarly correspondence, email. Not suitable for print (except low-budget short-run projects that require scholarly precision).  
  
## Method 4. Use a little HTML

HTML has access to the extended ASCII character set, which includes many accented non-English European vowels (umlaut, circumflex, etc.), some of which can serve as reasonable stand-ins for the long Pali vowels (_ä ï ü_ ; _à ì ù_ ; or _â î û_ etc.). The palatal _n_ is straightforward: _ñ._ Whatever type of accent you adopt, use it consistently.

Examples: |  pâ.nâtipâtâ verama.nî sikkhâ-pada.m samâdiyâmi   
(HTML: p&acirc;.n&acirc;tip&acirc;t&acirc; verama.n&icirc; sikkh&acirc;-pada.m sam&acirc;diy&acirc;mi) itihidam âyasmato kondaññassa, añña-kondañño'tveva nâmam, ahosîti   
(HTML: itihidam &acirc;yasmato konda&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-konda&ntilde;&ntilde;o'tveva n&acirc;mam, ahos&icirc;ti)  
---|---  
Readability: | Very good.  
Phonetics: | Fair. The consonantal diacritics are missing.  
Ease of use: | Good. Easy to produce using most HTML authoring tools.  
Portability: | Good. Limited to web browsers and other HTML-savvy software.  
Overall: | Fair-Good. Improves upon the capital letter method, but doesn't capture the consonantal diacritics.  
Uses: | Informal correspondence, email, print.  
  
## Method 5. Mixed Velthuis/HTML

This method attempts to clear up the stuttering of Method 3's doubled vowels, by using a little HTML (Method 4).

Examples: |  pâ.nâtipâtâ verama.nî sikkhâ-pada.m samâdiyâmi   
(HTML: p&acirc;.n&acirc;tip&acirc;t&acirc; verama.n&icirc; sikkh&acirc;-pada.m sam&acirc;diy&acirc;mi) itihida.m âyasmato ko.n.daññassa, añña-ko.n.dañño'tveva nâma.m, ahosîti   
(HTML: itihida.m &acirc;yasmato ko.n.da&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-ko.n.da&ntilde;&ntilde;o'tveva n&acirc;ma.m, ahos&icirc;ti)  
---|---  
Readability: | Fair. It looks like it has typos, although perhaps not quite as many as pure Velthuis.  
Phonetics: | Excellent.  
Ease of use: | Fair. More complex than Velthuis, since it requires a combination of special punctuation _and_ the use of special HTML characters.  
Portability: | Good. Limited to web browsers and other HTML-savvy software.  
Overall: | Fair. Although this hybrid does slightly improve the appearance of Velthuis, it still looks like an error-filled jumble.  
Uses: | Informal correspondence (scholars who demand precision are bound to prefer good old pure Velthuis). Not generally suitable for e-mail or print.  
  
## Method 6. Special Pali fonts

For high-quality print projects, nothing beats a well-designed Pali font. For an extensive review of available Pali and Sanskrit fonts, see [Transliteration and Devanagari Fonts for Sanskrit](<http://www.sanskritweb.net/fonts/> "Visit www.sanskritweb.net"), by Ulrich Stiehl. The Association for Insight Meditation's [Pali Font Resources page](<http://www.aimwell.org/Fonts/fonts.html> "Visit www.aimwell.org") offers several ANSI and Unicode fonts suitable for working with Pali.

Example (in "Normyn" font): |   
---|---  
Readability: | Excellent.  
Phonetics: | Excellent.  
Ease of use: | Variable — it depends on the keyboard mappings used by a particular font.  
Portability: | Poor. These fonts don't all share the same coding standards; they are not interchangeable. If I send you a text document that I formatted with font X, and you display it with font Y, the Pali characters may not show up properly.  
Overall: | Excellent — but only for documents that are to be shared in print (hard copy) form or as PDF files or GIF images.  
Uses: | Printing. Not suitable for e-mail or the web, except when embedded in PDF files or GIF images.  
  
## Method 7. Unicode and Unicode fonts

[Unicode](<http://www.unicode.org/> "Visit www.unicode.org") has emerged in recent years as the international standard for representing characters from most of the world's alphabets. All the special characters we need for Pali transliteration may be found in Unicode's [Latin Extended-A](<http://www.unicode.org/charts/PDF/U0100.pdf> "Visit www.unicode.org"), and [Latin Extended Additional](<http://www.unicode.org/charts/PDF/U1E00.pdf> "Visit www.unicode.org") code charts. They can therefore be easily generated using HTML, provided that your web browser uses a Unicode-savvy font.

There are many Unicode fonts available that contain the characters needed for Pali.

The following table lists the HTML Unicode entities required to generate each of the special Pali characters. If your web browser supports Unicode, the characters appearing in the last column of the table should resemble those appearing the shaded column. If they do not match, then you may have to upgrade your web browser, install Unicode fonts on your computer, or both. For details about configuring your computer and browser to use Unicode, see the [Unicode website](<http://www.unicode.org> "Visit www.unicode.org").

Pali letter | Velthuis | HTML | Rendered on your browser as [3]  
---|---|---|---  
A macron |  | AA | &#256; | Ā  
| aa | &#257; | ā  
I macron |  | II | &#298; | Ī  
| ii | &#299; | ī  
U macron |  | UU | &#362; | Ū  
| uu | &#363; | ū  
N dot-over |  | "N | &#7748; | Ṅ  
| "n | &#7749; | ṅ  
M dot-under |  | .M | &#7746; | Ṃ  
| .m | &#7747; | ṃ  
N tilde |  | ~N | &Ntilde; | Ñ  
| ~n | &ntilde; | ñ  
T dot-under |  | .T | &#7788; | Ṭ  
| .t | &#7789; | ṭ  
D dot-under |  | .D | &#7692; | Ḍ  
| .d | &#7693; | ḍ  
N dot-under |  | .N | &#7750; | Ṇ  
| .n | &#7751; | ṇ  
L dot-under |  | .L | &#7734; | Ḷ  
| .l | &#7735; | ḷ  
Examples: |  pānātipātā veramaṅī sikkhā-padaṁ samādiyāmi  
(HTML: p&#257;n&#257;tip&#257;t&#257; verama&#7749;&#299; sikkh&#257;-pada&#7745; sam&#257;diy&#257;mi) itihidaṁ āyasmato Koṇḍaññassa, añña-koṇḍañño'tveva nāmaṁ, ahosīti  
(HTML: itihida&#7745; &#257;yasmato Ko&#7751;&#7693;a&ntilde;&ntilde;assa, a&ntilde;&ntilde;a-ko&#7751;&#7693;a&ntilde;&ntilde;o'tveva n&#257;ma&#7745;, ahos&#299;ti)  
---|---  
Readability: | Excellent  
Phonetics: | Excellent  
Ease of use: | Poor-Good, depending on the particular software you use (HTML authoring program, word processor, e-mail client, etc.).  
Portability: | Good-Excellent. Requires the installation of at least a basic set of Unicode fonts.  
Overall: | Good. Still a little cumbersome to use in some software apps, a shortcoming that will probably fade in the next few years.  
Uses: | Web, email (if email client permits easy typing of Pali characters), print (with well-crafted Unicode fonts).
  *[MLS]: Middle Length Sayings (I.B.Horner, tr.; PTS)
  *[MLDB]: Middle Length Dsicourses of the Buddha (Bh. Bodhi, tr.; Wisdom Publications)
