# ContriMix and HistAuGAN Model Zoo and Baselines
This zoo contains all the training checkpoints for the model trained in our paper, together with their performance.
## TCGA-based model
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Algorithm</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Seed</th>
<th valign="bottom">Per-GPU Batch size</th>
<th valign="bottom">num_attr_vectors</th>
<th valign="bottom">num mixing per image</th>
<th valign="bottom">attr cons weight</th>
<th valign="bottom">self recon weight</th>
<th valign="bottom">cont cons weight</th>
<th valign="bottom">cont corr weight</th>
<th valign="bottom">attr similarity weight</th>
<th valign="bottom">Training type</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: HistAuGan encoder training -->
<tr><td align="center">TCGA 2.5M</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">0</td>
<td align="center">30</td>
<td align="center">16</td>
<td align="center">3</td>
<td align="center">0.1</td>
<td align="center">0.6</td>
<td align="center">0.2</td>
<td align="center">0.05</td>
<td align="center">0.05</td>
<td align="center">Encoder only</td>
<td align="center"><a href="https://drive.google.com/file/d/1b-cy8SnTiCpbv2icYdwGk1jrrm64QGos/view?usp=sharing">model</a></td>
</tr>
</tbody></table>


## Camelyon-based models
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Algorithm</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Seed</th>
<th valign="bottom">Per-GPU Batch size</th>
<th valign="bottom">nz</th>
<th valign="bottom">crop size</th>
<th valign="bottom">lambda_cls</th>
<th valign="bottom">Training type</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: HistAuGan encoder training -->
 <tr> <td align="center">Camelyon WILDS 17</td>
<td align="center">HistAuGAN</td>
<td align="center">50</td>
<td align="center">0</td>
<td align="center">30</td>
<td align="center">8</td>
<td align="center">216</td>
<td align="center">1.0</td>
<td align="center">Encoder only</td>
<td align="center"><a href="https://drive.google.com/file/d/1rUCSYjLIIh-H_8iLgw3eYQcqOPubpOir/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

