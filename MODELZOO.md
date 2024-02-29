# ContriMix and HistAuGAN Model Zoo and Baselines
This zoo collects all the trained ContriMix models trained in our paper, together with their performance.
- If you want to use ContriMix for augmentation in your project and want the best image quality, we highly recommend you use the model trained from the TCGA data 2.5 million samples across multiple indications.
- The checkpoints trained from the patch-wise WILDS Camelyon data do not have comparative image quality compared to the TCGA one. This is because the resolution from the training data was lower. Also, we trained these models for a relatively small number of epochs (20). These checkpoints should be only used to evaluate the performance of the backbone. We will release the ContriMix model trained on WSI data when ready.
- We additionally post the ContriMix checkpoints trained from the RxRx1 data together with its performance on the RxRx1.

## How to read the tables
- The tables are organized in different datasets.
- Training types:
    - `Encoders only`: only the ContriMix encoders were trained
    - `Backbone only`: only the backbone was be trained.
    - `Jointly`: the ContriMix encoders are jointly trained with the backbone (Camelyon WILDS only).

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
<td align="center">50</td>
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
<td align="center">Backbone only</td>
<td align="center"><a href="https://drive.google.com/file/d/1rUCSYjLIIh-H_8iLgw3eYQcqOPubpOir/view?usp=sharing">model</a></td>
</tr>
</tbody></table>


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Algorithm</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Learning rate</th>
<th valign="bottom">Seed</th>
<th valign="bottom">Per-GPU Batch size</th>
<th valign="bottom">num_attr_vectors</th>
<th valign="bottom">num mixing per image</th>
<th valign="bottom">attr cons weight</th>
<th valign="bottom">self recon weight</th>
<th valign="bottom">cont cons weight</th>
<th valign="bottom">entropy weight</th>
<th valign="bottom">Training type</th>
<th valign="bottom">Val. Acc. (%)</th>
<th valign="bottom">Test Acc. (%)</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: HistAuGan encoder training -->
<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">0</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">92.1</td>
<td align="center">95.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1k0_eDyiSl96XVvdTdghdUGS_0jRio0zg/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">1</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">91.8</td>
<td align="center">94.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1CdFJTUmwGniiIyZxx1ScYqKMr3vKZm1A/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">2</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">91.9</td>
<td align="center">95.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1CT-P31f1Vo36zHZaaAajd5J1tBAEtWCA/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">3</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">90.7</td>
<td align="center">95.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1YFQGYC4jHssZETGFkU709hBfd-poY8vD/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">4</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">92.5</td>
<td align="center">94.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1iKW6qf7YKXOXR53r5lieH9rpenJAeaT2/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">5</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">91.2</td>
<td align="center">94.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1Kx0C9J7eIMmqSokOO4jIImHsl4hNIrbY/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">6</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">92.6</td>
<td align="center">94.1</td>
<td align="center"><a href="https://drive.google.com/file/d/158NY_SXdu70Zi1fObwnN_qV1Xdk_CPLZ/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">7</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">91.9</td>
<td align="center">95.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1-Dx_ahuKP2bHfNk6inzb8A9v-ri4hfiR/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">8</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">91.3</td>
<td align="center">94.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1lV0Zt69fttUAYXWZ6RaRlGxvbeHBHml1/view?usp=sharing">model</a></td>
</tr>

<tr><td align="center">Camelyon WILDS</td>
<td align="center">Contrimix</td>
<td align="center">20</td>
<td align="center">1e-3</td>
<td align="center">9</td>
<td align="center">210</td>
<td align="center">4</td>
<td align="center">4</td>
<td align="center">0.1</td>
<td align="center">0.1</td>
<td align="center">0.3</td>
<td align="center">0.5</td>
<td align="center">Jointly</td>
<td align="center">92.1</td>
<td align="center">91.5</td>
<td align="center"><a href="https://drive.google.com/file/d/12RzqsM0KsvQjAZv53lDhRdY0138mIItj/view?usp=sharing">model</a></td>
</tr>

</tbody></table>

